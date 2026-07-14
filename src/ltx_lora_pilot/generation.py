from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable


ENDPOINTS = {
    "t2v-lora": "fal-ai/ltx-2.3-22b/distilled/text-to-video/lora",
    "t2v-base": "fal-ai/ltx-2.3-22b/distilled/text-to-video",
    "i2v-lora": "fal-ai/ltx-2.3-22b/distilled/image-to-video/lora",
    "i2v-base": "fal-ai/ltx-2.3-22b/distilled/image-to-video",
    "a2v-lora": "fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora",
}

NEGATIVE_PROMPT = (
    "different person, multiple people, duplicate person, distorted face, deformed facial features, "
    "asymmetrical eyes, bad hands, extra fingers, jitter, flicker, camera shake, text, subtitles, watermark, logo"
)


def build_generation_request(
    *,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    lora_url: str | None = None,
    image_url: str | None = None,
    audio_url: str | None = None,
    lora_scale: float = 0.8,
    seed: int | None = None,
    generate_audio: bool = True,
) -> tuple[str, dict]:
    if mode not in ENDPOINTS:
        raise ValueError(f"unknown generation mode: {mode}")
    if min(width, height, frames) <= 0 or width % 32 or height % 32 or frames % 8 != 1:
        raise ValueError("width and height must be positive multiples of 32 and frames must satisfy frames % 8 == 1")
    if mode.endswith("-lora") and not lora_url:
        raise ValueError("LoRA mode requires a lora URL")
    if mode.startswith("i2v") and not image_url:
        raise ValueError("image-to-video mode requires an image URL")
    if mode == "a2v-lora" and not audio_url:
        raise ValueError("audio-to-video mode requires an audio URL")

    payload = {
        "prompt": prompt,
        "num_frames": frames,
        "video_size": {"width": width, "height": height},
        "use_multiscale": True,
        "fps": 24,
        "scheduler": "ltx2",
        "acceleration": "none",
        "camera_lora": "static",
        "camera_lora_scale": 0.5,
        "negative_prompt": NEGATIVE_PROMPT,
        "enable_prompt_expansion": False,
        "enable_safety_checker": True,
        "video_output_type": "X264 (.mp4)",
        "video_quality": "high",
        "video_write_mode": "balanced",
    }
    if mode != "a2v-lora":
        payload["generate_audio"] = generate_audio
    else:
        payload["audio_url"] = audio_url
        payload["match_audio_length"] = False
    if image_url:
        payload["image_url"] = image_url
    if mode.endswith("-lora"):
        payload["loras"] = [{"path": lora_url, "scale": lora_scale}]
        payload["distill_lora_second_pass_scale"] = 0.5
    if seed is not None:
        payload["seed"] = seed
    return ENDPOINTS[mode], payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_uploaded_asset(path: Path, cache_path: Path, upload_fn: Callable[[Path], str]) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    asset_hash = _sha256(path)
    data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"assets": {}}
    cached = data["assets"].get(asset_hash)
    if cached:
        return cached["url"]

    url = upload_fn(path)
    data["assets"][asset_hash] = {"file_name": path.name, "bytes": path.stat().st_size, "url": url}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="assets-", suffix=".json", dir=cache_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, cache_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return url
