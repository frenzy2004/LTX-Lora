from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))

from run_ic_lora_provider import (  # noqa: E402
    IC_LORA_APPLICATION,
    IC_LORA_QUEUE_URL,
    ProviderExecutionError,
    build_debug_input,
    extract_unique_fal_key,
    release_unsubmitted_budget,
    reserve_budget_file,
    submit_once,
)


def test_extract_unique_fal_key_accepts_duplicate_mentions_of_same_key(
    tmp_path: Path,
) -> None:
    key = "12345678-1234-1234-1234-123456789abc:" + "a" * 32
    attachment = tmp_path / "attachment.txt"
    attachment.write_text(f"first {key}\nquoted again {key}\n", encoding="utf-8")
    assert extract_unique_fal_key(attachment) == key


def test_extract_unique_fal_key_rejects_ambiguity(tmp_path: Path) -> None:
    attachment = tmp_path / "attachment.txt"
    attachment.write_text(
        "12345678-1234-1234-1234-123456789abc:" + "a" * 32 + "\n"
        "87654321-4321-4321-4321-cba987654321:" + "b" * 32,
        encoding="utf-8",
    )
    with pytest.raises(ProviderExecutionError, match="exactly one unique"):
        extract_unique_fal_key(attachment)


def test_debug_input_is_fixed_to_reviewed_high_portrait_dataset() -> None:
    body = build_debug_input("https://storage.example/training.zip", steps=100)
    assert body == {
        "training_data_url": "https://storage.example/training.zip",
        "rank": 32,
        "number_of_steps": 100,
        "learning_rate": 0.0002,
        "number_of_frames": 89,
        "frame_rate": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        # Captions already contain SUBJECTX exactly once. Leaving this blank avoids
        # Fal prepending a duplicate trigger token during preprocessing.
        "trigger_phrase": "",
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "debug_dataset": True,
        "first_frame_conditioning_p": 0.1,
        "validation": [],
        "reference_downscale_factor": 1,
        "reference_temporal_scale_factor": 1,
    }


def test_budget_file_is_reserved_atomically_and_duplicate_label_is_blocked(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "incremental_absolute_stop": 40.0,
                "incremental_normal_cap": 40.0,
                "incremental_accounted_or_reserved": 0.0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    updated = reserve_budget_file(path, "ic_lora_provider_debug_100", 0.59)
    assert updated["incremental_accounted_or_reserved"] == 0.59
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["entries"][0]["status"] == "reserved"
    assert not path.with_suffix(".json.tmp").exists()

    with pytest.raises(ProviderExecutionError, match="already exists"):
        reserve_budget_file(path, "ic_lora_provider_debug_100", 0.59)


def test_unsubmitted_reservation_can_be_released_with_evidence(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "incremental_absolute_stop": 40.0,
                "incremental_normal_cap": 40.0,
                "incremental_accounted_or_reserved": 1.38,
                "incremental_remaining_absolute": 38.62,
                "incremental_remaining_normal_cap": 38.62,
                "entries": [
                    {"label": "completed", "amount_usd": 1.0, "status": "charged_expected"},
                    {"label": "upload-timeout", "amount_usd": 0.38, "status": "reserved"},
                ],
            }
        ),
        encoding="utf-8",
    )

    updated = release_unsubmitted_budget(
        path, "upload-timeout", "upload timed out before request acknowledgement"
    )
    assert updated["incremental_accounted_or_reserved"] == pytest.approx(1.0)
    assert updated["incremental_remaining_absolute"] == pytest.approx(39.0)
    assert updated["entries"][1]["status"] == "released_unsubmitted"
    assert "release_evidence" in updated["entries"][1]

    with pytest.raises(ProviderExecutionError, match="not an unsubmitted reservation"):
        release_unsubmitted_budget(path, "completed", "invalid release attempt")


def test_submit_once_uses_fixed_queue_and_no_retry_headers() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"request_id": "private-request-id"})

    result = submit_once(
        IC_LORA_APPLICATION,
        {"training_data_url": "https://storage.example/training.zip"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )
    assert result == {"request_id": "private-request-id"}
    assert seen["url"] == IC_LORA_QUEUE_URL
    headers = seen["headers"]
    assert headers["x-fal-no-retry"] == "1"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert headers["x-fal-store-io"] == "0"
    assert headers["authorization"] == "Key secret-key"


def test_submit_once_does_not_follow_redirects_or_accept_unknown_application() -> None:
    redirect_transport = httpx.MockTransport(
        lambda _request: httpx.Response(307, headers={"location": "https://bad.example"})
    )
    with pytest.raises(ProviderExecutionError, match="non-success"):
        submit_once(
            IC_LORA_APPLICATION,
            {"training_data_url": "https://storage.example/training.zip"},
            "secret-key",
            transport=redirect_transport,
        )
    with pytest.raises(ValueError, match="fixed"):
        submit_once(
            "fal-ai/not-the-approved-endpoint",
            {"training_data_url": "https://storage.example/training.zip"},
            "secret-key",
        )
