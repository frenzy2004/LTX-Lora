from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))

from run_ltx_v2v_inference import (  # noqa: E402
    LTX_V2V_APPLICATION,
    LTX_V2V_QUEUE_URL,
    build_parser,
    build_inference_input,
    estimate_cost,
    submit_inference_once,
    upload_files,
)


def test_build_inference_input_uses_rgb_reference_direct_control_and_max_quality() -> None:
    body = build_inference_input(
        video_url="https://storage.example/rgb-reference.mp4",
        control_video_url="https://storage.example/control.mp4",
        lora_url="https://storage.example/model.safetensors",
        prompt="SUBJECTX speaks naturally in a real office.",
        seed=12345,
        lora_scale=1.0,
    )
    assert body["video_url"] == "https://storage.example/rgb-reference.mp4"
    assert body["control_video_url"] == "https://storage.example/control.mp4"
    assert body["skip_control_preprocess"] is True
    assert body["preserve_original_video"] is False
    assert body["num_frames"] == 89
    assert body["resolution"] == "auto"
    assert body["frames_per_second"] == 24
    assert body["num_inference_steps"] == 30
    assert body["generate_audio"] is False
    assert body["enable_prompt_expansion"] is False
    assert body["video_quality"] == "maximum"
    assert body["loras"] == [
        {
            "path": "https://storage.example/model.safetensors",
            "scale": 1.0,
            "transformer": "both",
        }
    ]
    assert "AI-generated" in body["negative_prompt"]


def test_build_inference_input_accepts_optional_non_holdout_anchor() -> None:
    body = build_inference_input(
        video_url="https://storage.example/control.mp4",
        control_video_url="https://storage.example/control.mp4",
        lora_url="https://storage.example/model.safetensors",
        prompt="SUBJECTX speaks naturally.",
        seed=1,
        lora_scale=0.8,
        image_url="https://storage.example/reference.png",
    )
    assert body["image_url"] == "https://storage.example/reference.png"
    assert body["loras"][0]["scale"] == 0.8


def test_build_inference_input_accepts_long_valid_ltx_frame_count() -> None:
    body = build_inference_input(
        video_url="https://storage.example/control.mp4",
        control_video_url="https://storage.example/control.mp4",
        lora_url="https://storage.example/model.safetensors",
        prompt="SUBJECTX speaks the supplied script naturally.",
        seed=24071912,
        lora_scale=1.0,
        num_frames=265,
    )
    assert body["num_frames"] == 265


def test_build_inference_input_can_preserve_photorealistic_source_pixels() -> None:
    body = build_inference_input(
        video_url="https://storage.example/rgb-reference.mp4",
        control_video_url="https://storage.example/control.mp4",
        lora_url="https://storage.example/model.safetensors",
        prompt="SUBJECTX speaks naturally.",
        seed=24071943,
        lora_scale=0.6,
        num_frames=265,
        preserve_original_video=True,
        strength=0.35,
        video_strength=1.0,
    )
    assert body["preserve_original_video"] is True
    assert body["strength"] == pytest.approx(0.35)
    assert body["video_strength"] == pytest.approx(1.0)
    assert body["loras"][0]["scale"] == pytest.approx(0.6)


def test_build_inference_input_rejects_invalid_ltx_frame_count() -> None:
    with pytest.raises(ValueError, match="8n\\+1"):
        build_inference_input(
            video_url="https://storage.example/control.mp4",
            control_video_url="https://storage.example/control.mp4",
            lora_url="https://storage.example/model.safetensors",
            prompt="SUBJECTX speaks naturally.",
            seed=1,
            lora_scale=1.0,
            num_frames=262,
        )


def test_cost_estimate_uses_current_quality_megapixel_frame_rate() -> None:
    assert estimate_cost(544, 960, 89) == pytest.approx(0.1258428672)


def test_submit_inference_once_uses_fixed_endpoint_and_no_retry_headers() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"request_id": "private-inference-id"})

    result = submit_inference_once(
        LTX_V2V_APPLICATION,
        {"prompt": "test"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )
    assert result == {"request_id": "private-inference-id"}
    assert seen["url"] == LTX_V2V_QUEUE_URL
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
            LTX_V2V_APPLICATION,
            {"prompt": "test"},
            "secret-key",
            transport=transport,
        )


def test_upload_files_reuses_one_authenticated_client(tmp_path: Path) -> None:
    control = tmp_path / "control.mp4"
    lora = tmp_path / "model.safetensors"
    anchor = tmp_path / "anchor.png"
    for path in (control, lora, anchor):
        path.write_bytes(path.name.encode("utf-8"))

    created_clients: list[object] = []

    class FakeClient:
        def __init__(self) -> None:
            self.uploaded: list[Path] = []

        def upload_file(self, path: Path) -> str:
            self.uploaded.append(path)
            return f"https://storage.example/{path.name}"

    def client_factory(_key: str) -> FakeClient:
        client = FakeClient()
        created_clients.append(client)
        return client

    urls = upload_files(
        {"control": control, "lora": lora, "anchor": anchor},
        "secret-key",
        client_factory=client_factory,
    )

    assert len(created_clients) == 1
    assert urls == {
        "control": "https://storage.example/control.mp4",
        "lora": "https://storage.example/model.safetensors",
        "anchor": "https://storage.example/anchor.png",
    }


def test_start_parser_requires_separate_rgb_reference_video() -> None:
    args = build_parser().parse_args(
        [
            "start",
            "--video",
            "rgb-reference.mp4",
            "--control",
            "control.mp4",
            "--lora",
            "model.safetensors",
            "--audio",
            "audio.wav",
            "--expected-audio-sha256",
            "a" * 64,
            "--prompt",
            "SUBJECTX speaks directly to camera.",
            "--seed",
            "1",
            "--budget",
            "budget.json",
            "--key-source",
            "key.txt",
            "--state-dir",
            "state",
            "--label",
            "test",
        ]
    )
    assert args.video == Path("rgb-reference.mp4")
    assert args.expected_audio_sha256 == "a" * 64
