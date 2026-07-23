from __future__ import annotations

from decimal import Decimal, ROUND_UP
from typing import Any


TRIGGER = "orvo"
LTX23_T2V_TRAINER_ENDPOINT = "fal-ai/ltx23-trainer-v2/t2v"
LTX23_QUALITY_T2V_LORA_ENDPOINT = "fal-ai/ltx-2.3-quality/text-to-video/lora"
LTX23_QUALITY_I2V_LORA_ENDPOINT = "fal-ai/ltx-2.3-quality/image-to-video/lora"

TRAINING_RATE_PER_STEP = Decimal("0.006")
QUALITY_1080P_RATE_PER_SECOND = Decimal("0.06")

QUALITY_NEGATIVE_PROMPT = (
    "different person, multiple people, duplicate person, distorted face, deformed facial features, "
    "asymmetrical eyes, bad hands, extra fingers, jitter, flicker, camera shake, subtitles, watermark, logo, "
    "silent or muted audio, distorted voice, robotic voice, off-sync audio, added dialogue"
)

VALIDATION_PROMPTS = [
    'orvo says, "I think the real question is whether a sandwich can count as architecture."',
    (
        'orvo leans toward the camera and says, '
        '"Today I learned that coffee mugs have stronger opinions than most meetings."'
    ),
]


def money(value: Decimal | str | float | int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_UP)


def estimate_ltx23_t2v_training_cost(steps: int) -> Decimal:
    if steps < 100:
        raise ValueError("fal bills at least 100 training steps")
    return money(TRAINING_RATE_PER_STEP * steps)


def estimate_ltx23_quality_inference_cost(seconds: Decimal | float | int, resolution: str) -> Decimal:
    if resolution != "1080p":
        raise ValueError("only 1080p quality inference is costed for this pilot")
    duration = Decimal(str(seconds))
    if duration <= 0:
        raise ValueError("seconds must be positive")
    return money(QUALITY_1080P_RATE_PER_SECOND * duration)


def build_ltx23_t2v_training_payload(
    *,
    training_data_url: str,
    steps: int = 2000,
    rank: int = 32,
    learning_rate: float = 0.0002,
    number_of_frames: int = 121,
) -> dict[str, Any]:
    if not training_data_url:
        raise ValueError("training_data_url is required")
    if number_of_frames % 8 != 1:
        raise ValueError("number_of_frames must satisfy frames % 8 == 1")
    return {
        "training_data_url": training_data_url,
        "rank": rank,
        "number_of_steps": steps,
        "learning_rate": learning_rate,
        "number_of_frames": number_of_frames,
        "frame_rate": 24,
        "resolution": "medium",
        "aspect_ratio": "9:16",
        "trigger_phrase": TRIGGER,
        "auto_scale_input": True,
        "split_input_into_scenes": False,
        "with_audio": True,
        "audio_normalize": True,
        "audio_preserve_pitch": True,
        "validation": [{"prompt": prompt} for prompt in VALIDATION_PROMPTS],
        "validation_number_of_frames": 121,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
        "stg_scale": 1,
    }


def build_ltx23_quality_lora_payload(
    *,
    mode: str,
    prompt: str,
    lora_url: str,
    image_url: str | None = None,
    seed: int | None = None,
    lora_scale: float = 1.0,
    num_frames: int = 121,
) -> tuple[str, dict[str, Any]]:
    if mode not in {"t2v", "i2v"}:
        raise ValueError("mode must be 't2v' or 'i2v'")
    if not prompt:
        raise ValueError("prompt is required")
    if not lora_url:
        raise ValueError("lora_url is required")
    if num_frames % 8 != 1:
        raise ValueError("num_frames must satisfy frames % 8 == 1")
    if mode == "i2v" and not image_url:
        raise ValueError("image_url is required for i2v")

    endpoint = LTX23_QUALITY_T2V_LORA_ENDPOINT if mode == "t2v" else LTX23_QUALITY_I2V_LORA_ENDPOINT
    payload: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "resolution": "portrait_16_9",
        "frames_per_second": 24,
        "num_inference_steps": 15,
        "guidance_scale": 1,
        "generate_audio": True,
        "negative_prompt": QUALITY_NEGATIVE_PROMPT,
        "enable_prompt_expansion": False,
        "enable_safety_checker": True,
        "video_quality": "high",
        "video_write_mode": "balanced",
        "loras": [{"path": lora_url, "scale": lora_scale, "transformer": "both"}],
    }
    if image_url:
        payload["image_url"] = image_url
    if seed is not None:
        payload["seed"] = seed
    return endpoint, payload
