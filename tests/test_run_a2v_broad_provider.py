from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))


def test_expected_cost_is_derived_from_the_only_two_allowed_step_counts() -> None:
    from run_a2v_broad_provider import expected_training_cost

    assert expected_training_cost(100) == Decimal("0.6000")
    assert expected_training_cost(4_000) == Decimal("24.0000")
    with pytest.raises(ValueError, match="100 or 4000"):
        expected_training_cost(1_000)


def test_training_input_is_fixed_to_the_approved_a2v_contract() -> None:
    from run_a2v_broad_provider import build_training_input

    validation = [
        {
            "prompt": "A held-out person speaks naturally to the camera.",
            "image_url": "https://storage.example/holdout.png",
            "audio_url": "https://storage.example/holdout.wav",
        }
    ]
    body = build_training_input(
        "https://storage.example/training.zip",
        steps=4_000,
        trigger_phrase="subject_token_42",
        validation=validation,
    )

    assert body == {
        "training_data_url": "https://storage.example/training.zip",
        "rank": 32,
        "number_of_steps": 4_000,
        "learning_rate": 0.0001,
        "number_of_frames": 89,
        "frame_rate": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        "trigger_phrase": "subject_token_42",
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "debug_dataset": True,
        "audio_normalize": True,
        "audio_preserve_pitch": True,
        "validation": validation,
        "validation_negative_prompt": "distorted face, deformed mouth, extra teeth, identity drift, flicker, text, watermark",
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
    }


def test_training_input_rejects_unapproved_steps_or_nonsecure_urls() -> None:
    from run_a2v_broad_provider import build_training_input

    with pytest.raises(ValueError, match="100 or 4000"):
        build_training_input(
            "https://storage.example/training.zip",
            steps=1_000,
            trigger_phrase="subject_token_42",
            validation=[],
        )
    with pytest.raises(ValueError, match="secure"):
        build_training_input(
            "http://storage.example/training.zip",
            steps=100,
            trigger_phrase="subject_token_42",
            validation=[],
        )


def test_budget_guard_blocks_any_existing_paid_request_in_flight() -> None:
    from run_a2v_broad_provider import (
        ProviderExecutionError,
        assert_no_paid_request_in_flight,
    )

    assert_no_paid_request_in_flight(
        {
            "entries": [
                {"label": "done", "status": "charged_expected"},
                {"label": "released", "status": "released_unsubmitted"},
            ]
        }
    )
    for status in ("reserved", "uploaded", "submit_intent", "submitted"):
        with pytest.raises(ProviderExecutionError, match="in flight"):
            assert_no_paid_request_in_flight(
                {"entries": [{"label": "active", "status": status}]}
            )


def test_submit_once_uses_fixed_a2v_queue_and_disables_retries() -> None:
    from run_a2v_broad_provider import (
        A2V_APPLICATION,
        A2V_QUEUE_URL,
        submit_once,
    )

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"request_id": "private-request-id"})

    result = submit_once(
        A2V_APPLICATION,
        {"training_data_url": "https://storage.example/training.zip"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )

    assert result == {"request_id": "private-request-id"}
    assert seen["url"] == A2V_QUEUE_URL
    headers = seen["headers"]
    assert headers["x-fal-no-retry"] == "1"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert headers["x-fal-store-io"] == "0"
    assert headers["authorization"] == "Key secret-key"


def test_loss_extraction_preserves_structured_and_textual_provider_values() -> None:
    from run_a2v_broad_provider import extract_loss_observations

    snapshot = {
        "logs": [
            {"timestamp": "one", "message": "step 100 loss: 0.4821"},
            {"timestamp": "two", "training_loss": 0.4012},
            {"timestamp": "three", "message": "validation completed"},
        ]
    }

    observations = extract_loss_observations(snapshot)

    assert [item["value"] for item in observations] == [0.4821, 0.4012]
    assert observations[0]["source"] == "text"
    assert observations[1]["source"] == "field"


def test_price_evidence_must_be_current_official_and_exact_rate() -> None:
    from run_a2v_broad_provider import ProviderExecutionError, validate_price_evidence

    evidence = {
        "source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "rate_usd_per_step": "0.006",
        "response_sha256": "a" * 64,
        "retrieved_at_utc": "2026-07-20T00:00:00Z",
        "expires_at_utc": "2026-07-21T00:00:00Z",
    }
    assert validate_price_evidence(
        evidence,
        now=datetime(2026, 7, 20, 1, 0, tzinfo=timezone.utc),
    ) == evidence

    changed = dict(evidence, rate_usd_per_step="0.007")
    with pytest.raises(ProviderExecutionError, match="rate"):
        validate_price_evidence(
            changed,
            now=datetime(2026, 7, 20, 1, 0, tzinfo=timezone.utc),
        )


def test_budget_reservation_uses_derived_cost_and_blocks_second_request(
    tmp_path: Path,
) -> None:
    from run_a2v_broad_provider import (
        ProviderExecutionError,
        reserve_budget_file,
    )

    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "incremental_absolute_stop": 60.0,
                "incremental_normal_cap": 60.0,
                "incremental_accounted_or_reserved": 27.18,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    updated = reserve_budget_file(budget_path, "broad_a2v_debug_100", 100)

    assert updated["incremental_accounted_or_reserved"] == pytest.approx(27.78)
    assert updated["entries"][-1]["amount_usd"] == pytest.approx(0.6)
    assert updated["entries"][-1]["status"] == "reserved"
    with pytest.raises(ProviderExecutionError, match="in flight"):
        reserve_budget_file(budget_path, "broad_a2v_main_4000", 4_000)


def test_start_run_persists_submit_intent_before_the_only_post(tmp_path: Path) -> None:
    from run_a2v_broad_provider import start_run

    archive = tmp_path / "training.zip"
    archive.write_bytes(b"reviewed-archive")
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "incremental_absolute_stop": 60.0,
                "incremental_normal_cap": 60.0,
                "incremental_accounted_or_reserved": 27.18,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    key_source = tmp_path / "key.txt"
    key_source.write_text(
        "12345678-1234-1234-1234-123456789abc:" + "a" * 32,
        encoding="utf-8",
    )
    evidence = tmp_path / "price.json"
    evidence.write_text(
        json.dumps(
            {
                "source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
                "rate_usd_per_step": "0.006",
                "response_sha256": "b" * 64,
                "retrieved_at_utc": "2026-07-20T00:00:00Z",
                "expires_at_utc": "2026-07-21T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    post_count = 0

    def submit(_application: str, _arguments: dict, _key: str) -> dict[str, str]:
        nonlocal post_count
        post_count += 1
        state = json.loads((state_dir / "execution.private.json").read_text())
        assert state["phase"] == "submit_intent"
        return {"request_id": "private-request-id"}

    result = start_run(
        archive=archive,
        budget_path=budget,
        key_source=key_source,
        price_evidence_path=evidence,
        state_dir=state_dir,
        label="broad_a2v_debug_100",
        steps=100,
        trigger_phrase="subject_token_42",
        validation=[],
        upload_fn=lambda _path, _key: "https://storage.example/uploaded",
        submit_fn=submit,
        now=datetime(2026, 7, 20, 1, 0, tzinfo=timezone.utc),
    )

    assert post_count == 1
    assert result == {"phase": "submitted", "reserved_amount_usd": 0.6, "steps": 100}
    state = json.loads((state_dir / "execution.private.json").read_text())
    assert state["phase"] == "submitted"
    assert state["request_id"] == "private-request-id"
    persisted_budget = json.loads(budget.read_text())
    assert persisted_budget["entries"][-1]["status"] == "submitted"


def test_telemetry_snapshot_is_deduplicated_and_preserves_loss(tmp_path: Path) -> None:
    from run_a2v_broad_provider import record_telemetry_snapshot

    path = tmp_path / "telemetry.private.json"
    snapshot = {
        "status": "in_progress",
        "logs": [{"message": "step 400 loss=3.25e-1"}],
    }

    first = record_telemetry_snapshot(
        path,
        snapshot,
        observed_at_utc="2026-07-20T01:00:00Z",
    )
    second = record_telemetry_snapshot(
        path,
        snapshot,
        observed_at_utc="2026-07-20T01:01:00Z",
    )

    assert first["snapshot_count"] == 1
    assert second["snapshot_count"] == 1
    assert second["provider_loss_status"] == "exposed"
    assert second["loss_observations"][0]["value"] == pytest.approx(0.325)


def test_completed_monitor_records_missing_provider_loss_without_fabrication(
    tmp_path: Path,
) -> None:
    from run_a2v_broad_provider import monitor_run

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "execution.private.json").write_text(
        json.dumps(
            {
                "application": "fal-ai/ltx23-trainer-v2/a2v",
                "budget_label": "broad_a2v_debug_100",
                "phase": "submitted",
                "request_id": "private-request-id",
                "reserved_amount_usd": 0.6,
                "steps": 100,
            }
        ),
        encoding="utf-8",
    )
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "label": "broad_a2v_debug_100",
                        "amount_usd": 0.6,
                        "status": "submitted",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    key_source = tmp_path / "key.txt"
    key_source.write_text(
        "12345678-1234-1234-1234-123456789abc:" + "a" * 32,
        encoding="utf-8",
    )

    def download(_result: dict, destination: Path) -> list[str]:
        destination.mkdir(parents=True)
        names = ["lora_file.safetensors", "config_file.json", "debug_dataset.zip"]
        for name in names:
            (destination / name).write_bytes(name.encode())
        return names

    output = monitor_run(
        state_dir=state_dir,
        budget_path=budget,
        key_source=key_source,
        status_fn=lambda _key, _request_id: {
            "status": "completed",
            "logs": [{"message": "training completed"}],
        },
        result_fn=lambda _key, _request_id: {
            "lora_file": {"url": "https://storage.example/lora"},
            "config_file": {"url": "https://storage.example/config"},
            "debug_dataset": {"url": "https://storage.example/debug"},
        },
        download_fn=download,
        now=datetime(2026, 7, 20, 2, 0, tzinfo=timezone.utc),
    )

    assert output["phase"] == "completed"
    assert output["provider_loss_status"] == "provider_loss_not_exposed"
    state = json.loads((state_dir / "execution.private.json").read_text())
    assert state["provider_loss_status"] == "provider_loss_not_exposed"
    persisted_budget = json.loads(budget.read_text())
    assert persisted_budget["entries"][0]["status"] == "charged_expected"


def test_run_config_resolves_private_validation_inputs_from_its_directory(
    tmp_path: Path,
) -> None:
    from run_a2v_broad_provider import load_run_config

    (tmp_path / "holdout.png").write_bytes(b"png")
    (tmp_path / "holdout.wav").write_bytes(b"wav")
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "broad-a2v-provider-run-v1",
                "steps": 4_000,
                "trigger_phrase": "subject_token_42",
                "validation": [
                    {
                        "prompt": "A held-out person speaks naturally.",
                        "image": "holdout.png",
                        "audio": "holdout.wav",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_run_config(config)

    assert loaded["steps"] == 4_000
    assert loaded["validation"][0]["image"] == str(
        (tmp_path / "holdout.png").resolve()
    )
    assert loaded["validation"][0]["audio"] == str(
        (tmp_path / "holdout.wav").resolve()
    )


def test_completed_monitor_is_idempotent_without_another_provider_call(
    tmp_path: Path,
) -> None:
    from run_a2v_broad_provider import monitor_run

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "execution.private.json").write_text(
        json.dumps(
            {
                "application": "fal-ai/ltx23-trainer-v2/a2v",
                "phase": "completed",
                "provider_loss_status": "provider_loss_not_exposed",
                "provider_loss_observation_count": 0,
                "reserved_amount_usd": 24.0,
                "artifacts": {
                    "lora_file.safetensors": {"size_bytes": 1, "sha256": "a" * 64}
                },
            }
        ),
        encoding="utf-8",
    )

    output = monitor_run(
        state_dir=state_dir,
        budget_path=tmp_path / "unused-budget.json",
        key_source=tmp_path / "unused-key.txt",
        status_fn=lambda *_args: (_ for _ in ()).throw(
            AssertionError("provider status must not be called twice")
        ),
    )

    assert output["phase"] == "completed"
    assert output["provider_loss_status"] == "provider_loss_not_exposed"
