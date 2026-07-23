from decimal import Decimal

import pytest

from ltx_lora_pilot.ltx23_v2 import (
    estimate_ltx23_quality_inference_cost,
    estimate_ltx23_t2v_training_cost,
    build_ltx23_quality_lora_payload,
    build_ltx23_t2v_training_payload,
)


def test_training_cost_uses_current_fal_v2_t2v_rate() -> None:
    assert estimate_ltx23_t2v_training_cost(2000) == Decimal("12.0000")


def test_training_payload_uses_private_neutral_trigger() -> None:
    payload = build_ltx23_t2v_training_payload(training_data_url="https://private.invalid/training.zip")

    assert payload["trigger_phrase"] == "orvo"
    assert payload["number_of_steps"] == 2000
    assert payload["number_of_frames"] == 121
    assert payload["frame_rate"] == 24
    assert payload["aspect_ratio"] == "9:16"
    assert payload["with_audio"] is True
    assert "realname" not in str(payload).lower()
    assert "surname" not in str(payload).lower()


def test_training_payload_includes_validation_prompts() -> None:
    payload = build_ltx23_t2v_training_payload(training_data_url="https://private.invalid/training.zip")

    assert payload["validation"] == [
        {"prompt": 'orvo says, "I think the real question is whether a sandwich can count as architecture."'},
        {
            "prompt": (
                'orvo leans toward the camera and says, '
                '"Today I learned that coffee mugs have stronger opinions than most meetings."'
            )
        },
    ]
    assert payload["validation_number_of_frames"] == 121
    assert payload["validation_aspect_ratio"] == "9:16"


def test_quality_t2v_lora_payload_matches_fal_schema() -> None:
    endpoint, payload = build_ltx23_quality_lora_payload(
        mode="t2v",
        prompt='orvo says, "Coffee mugs have stronger opinions than most meetings."',
        lora_url="https://private.invalid/orvo.safetensors",
        seed=7,
    )

    assert endpoint == "fal-ai/ltx-2.3-quality/text-to-video/lora"
    assert payload["resolution"] == "portrait_16_9"
    assert payload["num_frames"] == 121
    assert payload["frames_per_second"] == 24
    assert payload["generate_audio"] is True
    assert payload["seed"] == 7
    assert payload["loras"] == [
        {"path": "https://private.invalid/orvo.safetensors", "scale": 1.0, "transformer": "both"}
    ]


def test_quality_i2v_requires_image_url() -> None:
    with pytest.raises(ValueError, match="image_url"):
        build_ltx23_quality_lora_payload(
            mode="i2v",
            prompt='orvo says, "This is a calm test."',
            lora_url="https://private.invalid/orvo.safetensors",
        )


def test_quality_inference_cost_is_conservative_for_five_second_1080p() -> None:
    assert estimate_ltx23_quality_inference_cost(5, "1080p") == Decimal("0.3000")
