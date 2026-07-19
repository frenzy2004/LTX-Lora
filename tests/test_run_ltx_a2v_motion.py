from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))

from run_ltx_a2v_motion import (  # noqa: E402
    LTX_A2V_APPLICATION,
    LTX_A2V_QUEUE_URL,
    build_a2v_input,
    estimate_cost,
    submit_inference_once,
    valid_frame_count_for_duration,
)


def test_valid_frame_count_rounds_up_to_ltx_8n_plus_1() -> None:
    assert valid_frame_count_for_duration(10.9, 24) == 265
    assert valid_frame_count_for_duration(3.708333, 24) == 89


def test_build_a2v_input_uses_first_frame_exact_audio_and_max_quality() -> None:
    body = build_a2v_input(
        audio_url="https://storage.example/speech.m4a",
        image_url="https://storage.example/reference.png",
        prompt="A real person speaks naturally to a phone camera.",
        seed=24071911,
    )
    assert body["match_audio_length"] is True
    assert body["resolution"] == "auto"
    assert body["frames_per_second"] == 24
    assert body["num_inference_steps"] == 30
    assert body["guidance_scale"] == 1.0
    assert body["generate_audio"] is True
    assert body["image_strength"] == 0.85
    assert body["enable_prompt_expansion"] is False
    assert body["video_quality"] == "maximum"
    assert "AI-generated" in body["negative_prompt"]


def test_cost_estimate_uses_current_quality_megapixel_frame_rate() -> None:
    assert estimate_cost(576, 960, 265) == pytest.approx(0.352781568)


def test_submit_inference_once_uses_fixed_endpoint_and_no_retry_headers() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"request_id": "private-a2v-id"})

    result = submit_inference_once(
        LTX_A2V_APPLICATION,
        {"prompt": "test"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )
    assert result == {"request_id": "private-a2v-id"}
    assert seen["url"] == LTX_A2V_QUEUE_URL
    headers = seen["headers"]
    assert headers["x-fal-no-retry"] == "1"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert headers["x-fal-store-io"] == "0"


def test_submit_inference_once_rejects_redirect() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(307, headers={"location": "https://bad.example"})
    )
    with pytest.raises(RuntimeError, match="non-success"):
        submit_inference_once(
            LTX_A2V_APPLICATION,
            {"prompt": "test"},
            "secret-key",
            transport=transport,
        )
