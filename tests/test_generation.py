from pathlib import Path

import pytest

from ltx_lora_pilot.generation import build_generation_request, resolve_uploaded_asset, validate_audio_input


def test_build_t2v_lora_request() -> None:
    endpoint, payload = build_generation_request(
        mode="t2v-lora",
        prompt="chrx9_person speaks in a studio",
        width=704,
        height=1248,
        frames=89,
        lora_url="https://private.invalid/adapter.safetensors",
        lora_scale=0.8,
        seed=42,
        generate_audio=True,
    )

    assert endpoint == "fal-ai/ltx-2.3-22b/distilled/text-to-video/lora"
    assert payload["video_size"] == {"width": 704, "height": 1248}
    assert payload["loras"] == [{"path": "https://private.invalid/adapter.safetensors", "scale": 0.8}]
    assert payload["num_frames"] == 89
    assert payload["generate_audio"] is True


def test_build_i2v_request_requires_image() -> None:
    with pytest.raises(ValueError, match="image"):
        build_generation_request(
            mode="i2v-lora",
            prompt="talking head",
            width=704,
            height=1248,
            frames=89,
            lora_url="https://private.invalid/adapter.safetensors",
        )


def test_build_a2v_request_requires_audio() -> None:
    with pytest.raises(ValueError, match="audio"):
        build_generation_request(
            mode="a2v-lora",
            prompt="talking head",
            width=704,
            height=1248,
            frames=265,
            lora_url="https://private.invalid/adapter.safetensors",
            image_url="https://private.invalid/reference.png",
        )


def test_validate_audio_input_rejects_video_container(tmp_path: Path) -> None:
    source = tmp_path / "voice.mp4"
    source.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Unsupported audio format"):
        validate_audio_input(source)


def test_validate_audio_input_accepts_wav(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"placeholder")

    validate_audio_input(source)


def test_resolve_uploaded_asset_reuses_private_cache(tmp_path: Path) -> None:
    asset = tmp_path / "adapter.bin"
    asset.write_bytes(b"adapter")
    cache = tmp_path / "assets.json"
    uploads = []

    def upload(path: Path) -> str:
        uploads.append(path)
        return "https://private.invalid/uploaded"

    first = resolve_uploaded_asset(asset, cache, upload)
    second = resolve_uploaded_asset(asset, cache, upload)

    assert first == second == "https://private.invalid/uploaded"
    assert uploads == [asset]
