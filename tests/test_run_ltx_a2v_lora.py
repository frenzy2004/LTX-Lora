from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))


def test_direct_a2v_lora_body_uses_audio_image_and_trained_lora() -> None:
    from run_ltx_a2v_lora import build_a2v_lora_input

    body = build_a2v_lora_input(
        audio_url="https://storage.example/complete-audio.wav",
        image_url="https://storage.example/real-first-frame.png",
        lora_url="https://storage.example/participant.safetensors",
        prompt="A real person speaks naturally to the camera.",
        seed=24071941,
        lora_scale=1.0,
    )

    assert body["audio_url"].endswith("complete-audio.wav")
    assert body["image_url"].endswith("real-first-frame.png")
    assert body["match_audio_length"] is True
    assert "num_frames" not in body
    assert body["frames_per_second"] == 24
    assert body["generate_audio"] is True
    assert body["num_inference_steps"] == 30
    assert body["video_quality"] == "maximum"
    assert body["loras"] == [
        {
            "path": "https://storage.example/participant.safetensors",
            "scale": 1.0,
            "transformer": "both",
        }
    ]


def test_complete_audio_hash_guard_accepts_only_the_explicit_source(
    tmp_path: Path,
) -> None:
    from run_ltx_a2v_lora import verify_expected_sha256

    audio = tmp_path / "complete-audio.wav"
    audio.write_bytes(b"the complete decoded whatsapp audio")
    expected = "13bedcae5acad870ee132a9b491dd1bc7dd491e04c1fdab4b9ee3bd4055f49cc"

    assert verify_expected_sha256(audio, expected) == expected.upper()
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_expected_sha256(audio, "0" * 64)


def test_valid_frame_count_and_cost_cover_the_complete_eleven_seconds() -> None:
    from run_ltx_a2v_lora import estimate_cost, valid_frame_count_for_duration

    assert valid_frame_count_for_duration(10.901333, 24) == 265
    assert estimate_cost(544, 960, 265) == pytest.approx(0.374700672)


def test_submit_is_fixed_to_ltx_a2v_lora_and_disables_retries() -> None:
    from run_ltx_a2v_lora import (
        LTX_A2V_LORA_APPLICATION,
        LTX_A2V_LORA_QUEUE_URL,
        submit_inference_once,
    )

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"request_id": "private-a2v-lora-id"})

    result = submit_inference_once(
        LTX_A2V_LORA_APPLICATION,
        {"prompt": "test"},
        "secret-key",
        transport=httpx.MockTransport(handler),
    )
    assert result == {"request_id": "private-a2v-lora-id"}
    assert seen["url"] == LTX_A2V_LORA_QUEUE_URL
    headers = seen["headers"]
    assert headers["x-fal-no-retry"] == "1"
    assert headers["x-app-fal-disable-fallback"] == "true"
    assert headers["x-fal-store-io"] == "0"


def test_submit_rejects_redirect_instead_of_following_it() -> None:
    from run_ltx_a2v_lora import LTX_A2V_LORA_APPLICATION, submit_inference_once

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            307, headers={"location": "https://unexpected.example"}
        )
    )
    with pytest.raises(RuntimeError, match="non-success"):
        submit_inference_once(
            LTX_A2V_LORA_APPLICATION,
            {"prompt": "test"},
            "secret-key",
            transport=transport,
        )


def test_upload_files_reuses_one_authenticated_client(tmp_path: Path) -> None:
    from run_ltx_a2v_lora import upload_files

    audio = tmp_path / "audio.wav"
    image = tmp_path / "image.png"
    lora = tmp_path / "model.safetensors"
    for path in (audio, image, lora):
        path.write_bytes(path.name.encode("utf-8"))

    created_clients: list[object] = []

    class FakeClient:
        def upload_file(self, path: Path) -> str:
            return f"https://storage.example/{path.name}"

    def client_factory(_key: str) -> FakeClient:
        client = FakeClient()
        created_clients.append(client)
        return client

    urls = upload_files(
        {"audio": audio, "image": image, "lora": lora},
        "secret-key",
        client_factory=client_factory,
    )
    assert len(created_clients) == 1
    assert urls == {
        "audio": "https://storage.example/audio.wav",
        "image": "https://storage.example/image.png",
        "lora": "https://storage.example/model.safetensors",
    }


def test_start_parser_requires_explicit_audio_hash_and_lora() -> None:
    from run_ltx_a2v_lora import build_parser

    args = build_parser().parse_args(
        [
            "start",
            "--audio",
            "complete.wav",
            "--expected-audio-sha256",
            "a" * 64,
            "--image",
            "real-frame.png",
            "--lora",
            "participant.safetensors",
            "--prompt",
            "SUBJECTX speaks naturally.",
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
    assert args.expected_audio_sha256 == "a" * 64
    assert args.lora == Path("participant.safetensors")
