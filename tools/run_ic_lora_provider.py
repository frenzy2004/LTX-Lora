from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from build_ic_lora_dataset import BudgetExceeded, reserve_budget


IC_LORA_APPLICATION = "fal-ai/ltx23-trainer-v2/ic-lora/v2v"
IC_LORA_QUEUE_URL = "https://queue.fal.run/fal-ai/ltx23-trainer-v2/ic-lora/v2v"
KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}:[0-9a-fA-F]{32})"
    r"(?![A-Za-z0-9])"
)


class ProviderExecutionError(RuntimeError):
    pass


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def extract_unique_fal_key(path: Path) -> str:
    if not path.is_file():
        raise ProviderExecutionError("credential attachment is unavailable")
    text = path.read_text(encoding="utf-8", errors="strict")
    matches = sorted(set(KEY_PATTERN.findall(text)))
    if len(matches) != 1:
        raise ProviderExecutionError(
            "credential attachment must contain exactly one unique Fal key"
        )
    return matches[0]


def build_debug_input(training_data_url: str, *, steps: int) -> dict[str, Any]:
    if not isinstance(training_data_url, str) or not training_data_url.startswith(
        "https://"
    ):
        raise ValueError("a secure provider training-data URL is required")
    if steps < 100 or steps > 20_000:
        raise ValueError("steps must be within the provider range 100..20000")
    return {
        "training_data_url": training_data_url,
        "rank": 32,
        "number_of_steps": steps,
        "learning_rate": 0.0002,
        "number_of_frames": 89,
        "frame_rate": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        # Every reviewed caption already contains SUBJECTX exactly once. Fal
        # prepends trigger_phrase, so leaving it blank prevents a duplicate token.
        "trigger_phrase": "",
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "debug_dataset": True,
        "first_frame_conditioning_p": 0.1,
        "validation": [],
        "reference_downscale_factor": 1,
        "reference_temporal_scale_factor": 1,
    }


def reserve_budget_file(path: Path, label: str, amount: float) -> dict[str, Any]:
    if not path.is_file():
        raise ProviderExecutionError("budget ledger is unavailable")
    budget = json.loads(path.read_text(encoding="utf-8"))
    if any(entry.get("label") == label for entry in budget.get("entries", [])):
        raise ProviderExecutionError(f"budget label already exists: {label}")
    try:
        updated = reserve_budget(budget, label, amount)
    except BudgetExceeded as exc:
        raise ProviderExecutionError(str(exc)) from exc
    atomic_write_json(path, updated)
    return updated


def update_budget_entry(
    path: Path, label: str, status: str, **extra: Any
) -> dict[str, Any]:
    budget = json.loads(path.read_text(encoding="utf-8"))
    matches = [entry for entry in budget.get("entries", []) if entry.get("label") == label]
    if len(matches) != 1:
        raise ProviderExecutionError("reserved budget entry is unavailable or ambiguous")
    matches[0]["status"] = status
    matches[0].update(extra)
    atomic_write_json(path, budget)
    return budget


def release_unsubmitted_budget(
    path: Path, label: str, evidence: str
) -> dict[str, Any]:
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("release evidence is required")
    budget = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        entry for entry in budget.get("entries", []) if entry.get("label") == label
    ]
    if len(matches) != 1:
        raise ProviderExecutionError("reserved budget entry is unavailable or ambiguous")
    entry = matches[0]
    if entry.get("status") != "reserved" or entry.get("submitted_at_utc"):
        raise ProviderExecutionError("budget entry is not an unsubmitted reservation")
    amount = Decimal(str(entry.get("amount_usd", 0)))
    current = Decimal(str(budget.get("incremental_accounted_or_reserved", 0)))
    updated_total = current - amount
    if updated_total < 0:
        raise ProviderExecutionError("budget release would make accounted spend negative")
    budget["incremental_accounted_or_reserved"] = float(updated_total)
    if "incremental_absolute_stop" in budget:
        absolute_cap = Decimal(str(budget["incremental_absolute_stop"]))
        budget["incremental_remaining_absolute"] = float(absolute_cap - updated_total)
    if "incremental_normal_cap" in budget:
        normal_cap = Decimal(str(budget["incremental_normal_cap"]))
        budget["incremental_remaining_normal_cap"] = float(normal_cap - updated_total)
    entry.update(
        {
            "status": "released_unsubmitted",
            "released_at_utc": utc_now(),
            "release_evidence": evidence.strip(),
        }
    )
    atomic_write_json(path, budget)
    return budget


def submit_once(
    application: str,
    arguments: dict[str, Any],
    key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    if application != IC_LORA_APPLICATION:
        raise ValueError("IC-LoRA queue endpoint is fixed")
    if not isinstance(key, str) or not key:
        raise ProviderExecutionError("Fal key is unavailable")
    body = canonical_json_bytes(arguments)
    selected_transport: httpx.BaseTransport = transport or httpx.HTTPTransport(
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
            response = client.post(IC_LORA_QUEUE_URL, content=body, headers=headers)
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
    request_id = acknowledgement.get("request_id") if isinstance(acknowledgement, dict) else None
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProviderExecutionError(
            "provider submission acknowledgement is malformed; do not retry"
        )
    return {"request_id": request_id}


def upload_archive(path: Path, key: str) -> str:
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ProviderExecutionError("reviewed training archive is unavailable")
    import fal_client

    url = fal_client.SyncClient(key=key, default_timeout=600.0).upload_file(path)
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ProviderExecutionError("provider upload did not return a secure URL")
    return url


def start_run(
    *,
    archive: Path,
    budget_path: Path,
    key_source: Path,
    state_dir: Path,
    label: str,
    steps: int,
    amount: float,
) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "execution.private.json"
    if state_path.exists():
        raise ProviderExecutionError("execution state already exists; refusing duplicate submit")

    archive_digest = sha256_file(archive)
    reserve_budget_file(budget_path, label, amount)
    state: dict[str, Any] = {
        "application": IC_LORA_APPLICATION,
        "archive_sha256": archive_digest,
        "archive_size_bytes": archive.stat().st_size,
        "budget_label": label,
        "reserved_amount_usd": amount,
        "steps": steps,
        "phase": "reserved",
        "created_at_utc": utc_now(),
    }
    atomic_write_json(state_path, state)

    key = extract_unique_fal_key(key_source)
    uploaded_url = upload_archive(archive, key)
    arguments = build_debug_input(uploaded_url, steps=steps)
    state.update(
        {
            "phase": "uploaded",
            "upload_url_sha256": hashlib.sha256(uploaded_url.encode("utf-8")).hexdigest(),
            "request_body_sha256": hashlib.sha256(
                canonical_json_bytes(arguments)
            ).hexdigest(),
            "uploaded_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)

    acknowledgement = submit_once(IC_LORA_APPLICATION, arguments, key)
    state.update(
        {
            "phase": "submitted",
            "request_id": acknowledgement["request_id"],
            "submitted_at_utc": utc_now(),
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
        "reserved_amount_usd": amount,
        "steps": steps,
        "archive_sha256": archive_digest,
    }


def _safe_artifact_name(field: str, value: dict[str, Any]) -> str:
    provider_name = value.get("file_name")
    suffix = Path(provider_name).suffix if isinstance(provider_name, str) else ""
    default_suffix = {
        "lora_file": ".safetensors",
        "config_file": ".json",
        "debug_dataset": ".zip",
        "video": ".mp4",
    }.get(field, ".bin")
    if not suffix or len(suffix) > 12:
        suffix = default_suffix
    return re.sub(r"[^A-Za-z0-9_.-]", "_", field) + suffix


def download_result_artifacts(result: dict[str, Any], destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for field in ("lora_file", "config_file", "debug_dataset", "video"):
        value = result.get(field)
        if not isinstance(value, dict):
            continue
        url = value.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        name = _safe_artifact_name(field, value)
        target = destination / name
        temporary = target.with_suffix(target.suffix + ".part")
        transport = httpx.HTTPTransport(retries=0, verify=True, trust_env=False)
        with httpx.Client(
            transport=transport,
            follow_redirects=True,
            trust_env=False,
            timeout=600.0,
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
        os.replace(temporary, target)
        downloaded.append(name)
    return downloaded


def monitor_run(
    *, state_dir: Path, budget_path: Path, key_source: Path
) -> dict[str, Any]:
    state_path = state_dir / "execution.private.json"
    if not state_path.is_file():
        raise ProviderExecutionError("execution state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request_id = state.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProviderExecutionError("execution has no submitted provider request")
    key = extract_unique_fal_key(key_source)
    import fal_client

    client = fal_client.SyncClient(key=key, default_timeout=120.0)
    status = client.status(IC_LORA_APPLICATION, request_id, with_logs=False)
    status_name = type(status).__name__.lower()
    state["last_provider_status"] = status_name
    state["last_checked_at_utc"] = utc_now()
    atomic_write_json(state_path, state)

    if status_name != "completed":
        return {
            "phase": state.get("phase"),
            "provider_status": status_name,
            "reserved_amount_usd": state.get("reserved_amount_usd"),
        }

    error = getattr(status, "error", None)
    if error:
        state["phase"] = "failed_pending_billing_verification"
        state["provider_error_type"] = getattr(status, "error_type", None)
        state["completed_at_utc"] = utc_now()
        atomic_write_json(state_path, state)
        update_budget_entry(
            budget_path,
            state["budget_label"],
            "failed_pending_billing_verification",
            completed_at_utc=state["completed_at_utc"],
        )
        return {
            "phase": state["phase"],
            "provider_status": status_name,
            "reserved_amount_usd": state.get("reserved_amount_usd"),
        }

    result = client.result(IC_LORA_APPLICATION, request_id)
    if not isinstance(result, dict):
        raise ProviderExecutionError("provider result has an unexpected shape")
    atomic_write_json(state_dir / "result.private.json", result)
    artifacts = download_result_artifacts(result, state_dir / "artifacts")
    required = {"lora_file", "config_file", "debug_dataset"}
    present = {name.split(".", 1)[0] for name in artifacts}
    missing = sorted(required - present)
    if missing:
        raise ProviderExecutionError(
            "completed provider result is missing required artifacts: " + ", ".join(missing)
        )

    state.update(
        {
            "phase": "completed",
            "completed_at_utc": utc_now(),
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
        state["budget_label"],
        "charged_expected",
        completed_at_utc=state["completed_at_utc"],
    )
    return {
        "phase": "completed",
        "provider_status": status_name,
        "reserved_amount_usd": state.get("reserved_amount_usd"),
        "artifacts": artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot Fal LTX IC-LoRA execution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--archive", type=Path, required=True)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--key-source", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--label", default="ic_lora_provider_debug_100")
    start.add_argument("--steps", type=int, default=100)
    start.add_argument("--amount", type=float, default=0.59)

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--budget", type=Path, required=True)
    monitor.add_argument("--key-source", type=Path, required=True)
    monitor.add_argument("--state-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "start":
        output = start_run(
            archive=args.archive,
            budget_path=args.budget,
            key_source=args.key_source,
            state_dir=args.state_dir,
            label=args.label,
            steps=args.steps,
            amount=args.amount,
        )
    else:
        output = monitor_run(
            state_dir=args.state_dir,
            budget_path=args.budget,
            key_source=args.key_source,
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
