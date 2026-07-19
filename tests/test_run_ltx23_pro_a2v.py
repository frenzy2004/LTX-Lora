from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))


def test_pro_a2v_body_uses_complete_audio_real_frame_and_portrait() -> None:
    from run_ltx23_pro_a2v import build_pro_a2v_input

    body = build_pro_a2v_input(
        audio_url="https://storage.example/complete.wav",
        image_url="https://storage.example/real-frame.png",
        prompt="The real person speaks naturally to camera.",
    )
    assert body == {
        "audio_url": "https://storage.example/complete.wav",
        "image_url": "https://storage.example/real-frame.png",
        "prompt": "The real person speaks naturally to camera.",
        "guidance_scale": 9.0,
        "aspect_ratio": "9:16",
    }


def test_pro_a2v_cost_is_current_per_second_rate() -> None:
    from run_ltx23_pro_a2v import estimate_cost

    assert estimate_cost(10.857333) == pytest.approx(1.0857333)


def test_pro_a2v_submit_is_fixed_and_no_retry() -> None:
    from run_ltx23_pro_a2v import (
        LTX23_PRO_A2V_APPLICATION,
        LTX23_PRO_A2V_QUEUE_URL,
        submit_inference_once,
    )

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"request_id": "private-pro-a2v-id"})

    result = submit_inference_once(
        LTX23_PRO_A2V_APPLICATION,
        {"audio_url": "https://storage.example/audio.wav"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )
    assert result == {"request_id": "private-pro-a2v-id"}
    assert seen["url"] == LTX23_PRO_A2V_QUEUE_URL
    headers = seen["headers"]
    assert headers["x-fal-no-retry"] == "1"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert headers["x-fal-store-io"] == "0"
