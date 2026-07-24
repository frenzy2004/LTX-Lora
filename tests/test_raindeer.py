from decimal import Decimal

from ltx_lora_pilot.raindeer import (
    DEFAULT_FIVE_SECOND_FRAMES,
    RenderSpec,
    generated_megapixels,
    proof_file,
    quality_render_cost,
    render_cost,
    round_cost,
    training_cost,
)


def test_raindeer_training_cost_uses_ltx23_trainer_steps() -> None:
    assert training_cost(500) == Decimal("3.0000")
    assert training_cost(1000) == Decimal("6.0000")


def test_raindeer_quality_render_cost_uses_generated_megapixel_frames() -> None:
    assert generated_megapixels(width=1280, height=720, frames=121) == Decimal("111.5136")
    assert quality_render_cost(width=1280, height=720, frames=121) == Decimal("0.3020")
    assert quality_render_cost(width=1280, height=720, frames=241) == Decimal("0.6014")


def test_raindeer_round_cost_adds_training_and_renders() -> None:
    renders = (
        RenderSpec("one", "t2v", frames=DEFAULT_FIVE_SECOND_FRAMES),
        RenderSpec("two", "t2v", frames=DEFAULT_FIVE_SECOND_FRAMES),
        RenderSpec("three", "i2v", frames=DEFAULT_FIVE_SECOND_FRAMES),
    )
    assert render_cost(renders[0]) == Decimal("0.3020")
    assert round_cost(type("Round", (), {"training_steps": 1000, "renders": renders})()) == Decimal("6.9060")


def test_raindeer_proof_file_hashes_without_private_metadata(tmp_path) -> None:
    proof_video = tmp_path / "raindeer-round-proof.mp4"
    proof_video.write_bytes(b"generated proof")

    entry = proof_file(proof_video, quality_status="round_proof")

    assert entry["filename"] == "raindeer-round-proof.mp4"
    assert entry["bytes"] == len(b"generated proof")
    assert entry["quality_status"] == "round_proof"
    assert entry["manual_review"] == {
        "output_classification": "generated_output",
        "consent_or_authorization": "confirmed",
        "embedded_source_asset_metadata": False,
    }
