from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import httpx
import pytest

import ltx_lora_pilot.a2v_static_verification as static_verification
import ltx_lora_pilot.a2v_execution as execution
from ltx_lora_pilot.a2v_execution import (
    AmbiguousProviderSubmission,
    execute_training_bundle,
)
from ltx_lora_pilot.fal_api import A2V_QUEUE_URL, submit_a2v_once
from ltx_lora_pilot.preflight import PreflightNotReady
from test_preflight import EXECUTION_ID, _clock, _typed_id, _write_ready_run


@pytest.fixture
def ready_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    fixture = _write_ready_run(tmp_path)

    monkeypatch.setattr(
        static_verification, "_WINDOWS_DACL_CHECK", lambda _path: None
    )

    def structural_validator(path: Path, **_: Any) -> dict[str, Any]:
        if Path(path) == fixture["run_dir"] / "candidates":
            return fixture["structural"]
        return fixture["train_report"]

    monkeypatch.setattr(
        static_verification, "validate_a2v_directory", structural_validator
    )
    return fixture


def _execute(
    fixture: dict[str, Any],
    *,
    resolve_key=lambda: "test-key",
    upload_fn=lambda path: f"https://private.invalid/{path.name}?opaque=1",
    submit_fn=lambda _endpoint, _body, _key: {"request_id": "request-test-001"},
):
    return execute_training_bundle(
        fixture["run_dir"],
        fixture["bundle_id"],
        approved_private_root=fixture["private_root"],
        resolve_key=resolve_key,
        upload_fn=upload_fn,
        submit_fn=submit_fn,
        clock=_clock,
    )


def _latest_state(ledger_path: Path) -> str:
    with sqlite3.connect(ledger_path) as connection:
        value = connection.execute(
            "SELECT to_state FROM events ORDER BY event_index DESC LIMIT 1"
        ).fetchone()
    assert value is not None
    return str(value[0])


def test_preflight_failure_never_reads_key_reserves_or_calls_provider(
    ready_run: dict[str, Any],
) -> None:
    calls: list[str] = []

    with pytest.raises(PreflightNotReady):
        execute_training_bundle(
            ready_run["run_dir"],
            "0" * 64,
            approved_private_root=ready_run["private_root"],
            resolve_key=lambda: calls.append("key") or "test-key",
            upload_fn=lambda _path: calls.append("upload") or "https://private.invalid/a",
            submit_fn=lambda *_args: calls.append("submit") or {"request_id": "x"},
            clock=_clock,
        )

    assert calls == []
    assert ready_run["ledger"].remaining() == Decimal("8.4591")


def test_changed_ledger_head_after_preflight_fails_before_key_or_upload(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_preflight = execution.run_preflight
    calls: list[str] = []

    def raced_preflight(*args: Any, **kwargs: Any):
        report = real_preflight(*args, **kwargs)
        ready_run["ledger"].reserve_training(
            "f" * 64,
            _typed_id("exec", 500),
            Decimal("0.0001"),
            expected_head_sha256=ready_run["ledger"].head_hash,
        )
        return report

    monkeypatch.setattr(execution, "run_preflight", raced_preflight)

    with pytest.raises(RuntimeError, match="ledger head"):
        _execute(
            ready_run,
            resolve_key=lambda: calls.append("key") or "test-key",
            upload_fn=lambda _path: calls.append("upload") or "https://private.invalid/a",
        )

    assert calls == []


def test_upload_failure_releases_the_training_reservation(ready_run: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="upload failed"):
        _execute(
            ready_run,
            upload_fn=lambda _path: (_ for _ in ()).throw(RuntimeError("upload failed")),
        )

    assert ready_run["ledger"].remaining() == Decimal("8.4591")
    assert _latest_state(ready_run["ledger"].path) == "released"


def test_mutation_after_upload_releases_before_submit(ready_run: dict[str, Any]) -> None:
    calls: list[str] = []

    def mutate_archive(path: Path) -> str:
        calls.append(path.name)
        if path.name == "training-data.zip":
            os.chmod(path, 0o600)
            path.write_bytes(path.read_bytes() + b"tampered")
        return f"https://private.invalid/{path.name}?opaque=1"

    # On Windows the retained CreateFileW guard blocks the write itself; on
    # platforms that permit the write, the post-upload digest check rejects it.
    with pytest.raises(RuntimeError, match="upload failed"):
        _execute(
            ready_run,
            upload_fn=mutate_archive,
            submit_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("must not submit")),
        )

    assert calls == ["training-data.zip", *calls[1:]]
    assert ready_run["ledger"].remaining() == Decimal("8.4591")
    assert _latest_state(ready_run["ledger"].path) == "released"


def test_submit_started_is_durable_before_one_exact_flat_submission(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_states: list[str] = []
    payloads: list[dict[str, Any]] = []
    uploaded: list[Path] = []

    def upload(path: Path) -> str:
        uploaded.append(path)
        return f"https://private.invalid/{path.name}?opaque=1"

    def submit(endpoint: str, body: bytes, _key: str) -> dict[str, str]:
        observed_states.append(_latest_state(ready_run["ledger"].path))
        assert endpoint == A2V_QUEUE_URL
        payloads.append(json.loads(body))
        return {"request_id": "request-test-001"}

    real_transition = execution.PilotLedger.transition

    def transition_with_durable_ack(
        ledger: execution.PilotLedger, reservation_id: str, to_state: str
    ) -> None:
        if to_state == "submitted":
            request_record = (
                Path(ready_run["private_root"])
                / ".a2v-provider-state"
                / f"{ready_run['bundle_id']}.request.json"
            )
            assert request_record.is_file()
            assert json.loads(request_record.read_text(encoding="utf-8"))["request_id"] == (
                "request-test-001"
            )
        real_transition(ledger, reservation_id, to_state)

    monkeypatch.setattr(execution.PilotLedger, "transition", transition_with_durable_ack)

    result = _execute(ready_run, upload_fn=upload, submit_fn=submit)

    assert observed_states == ["submit_started"]
    assert result.request_id == "request-test-001"
    assert ready_run["ledger"].state(result.reservation_id) == "submitted"
    payload = payloads[0]
    assert set(payload) == {
        "training_data_url",
        "rank",
        "number_of_steps",
        "learning_rate",
        "number_of_frames",
        "frame_rate",
        "resolution",
        "aspect_ratio",
        "trigger_phrase",
        "auto_scale_input",
        "split_input_into_scenes",
        "debug_dataset",
        "audio_normalize",
        "audio_preserve_pitch",
        "validation",
        "validation_negative_prompt",
        "validation_number_of_frames",
        "validation_frame_rate",
        "validation_resolution",
        "validation_aspect_ratio",
    }
    assert payload["number_of_steps"] == 1000
    assert len(payload["validation"]) == 2
    assert all(set(item) == {"prompt", "image_url", "audio_url"} for item in payload["validation"])
    assert len(uploaded) == 5
    assert all(path.parent.name == ready_run["bundle_id"] for path in uploaded)
    assert all("candidates" not in path.parts for path in uploaded)
    records = list((Path(ready_run["private_root"]) / ".a2v-provider-state").rglob("*.json"))
    assert records
    assert all("private.invalid" not in path.read_text(encoding="utf-8") for path in records)


def test_ambiguous_submit_is_not_retried_and_stays_committed(
    ready_run: dict[str, Any],
) -> None:
    attempts: list[int] = []

    def timeout_submit(*_args: Any) -> dict[str, str]:
        attempts.append(1)
        raise TimeoutError("request timed out")

    with pytest.raises(AmbiguousProviderSubmission, match="ambiguous"):
        _execute(ready_run, submit_fn=timeout_submit)

    assert attempts == [1]
    assert _latest_state(ready_run["ledger"].path) == "submit_started"
    assert ready_run["ledger"].remaining() == Decimal("2.4591")


def test_submit_a2v_once_uses_one_non_redirecting_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"request_id": "request-test-001"})

    result = submit_a2v_once(
        A2V_QUEUE_URL,
        b'{"rank":32}',
        key="test-key",
        transport=httpx.MockTransport(handler),
    )

    assert result == {"request_id": "request-test-001"}
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == A2V_QUEUE_URL
    assert requests[0].headers["Authorization"] == "Key test-key"
    assert requests[0].headers["Content-Type"] == "application/json"
    assert requests[0].headers["Accept"] == "application/json"
    assert requests[0].headers["X-Fal-No-Retry"] == "1"
    assert requests[0].headers["x-app-fal-disable-fallback"] == "true"
    assert requests[0].headers["X-Fal-Store-IO"] == "0"


def test_submit_a2v_once_accepts_the_executor_positional_key() -> None:
    result = submit_a2v_once(
        A2V_QUEUE_URL,
        b"{}",
        "test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"request_id": "request-test-002"})
        ),
    )

    assert result == {"request_id": "request-test-002"}


def test_submit_a2v_once_rejects_redirect_without_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.invalid"})

    with pytest.raises(RuntimeError, match="status"):
        submit_a2v_once(
            A2V_QUEUE_URL,
            b"{}",
            key="test-key",
            transport=httpx.MockTransport(handler),
        )

    assert len(requests) == 1
