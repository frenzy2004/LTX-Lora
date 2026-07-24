from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Any


TRAINING_RATE_PER_STEP = Decimal("0.006")
QUALITY_RATE_PER_MEGAPIXEL = Decimal("0.0027075")
DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720
DEFAULT_FIVE_SECOND_FRAMES = 121
DEFAULT_TEN_SECOND_FRAMES = 241


@dataclass(frozen=True)
class RenderSpec:
    name: str
    mode: str
    frames: int = DEFAULT_FIVE_SECOND_FRAMES
    width: int = DEFAULT_RENDER_WIDTH
    height: int = DEFAULT_RENDER_HEIGHT


@dataclass(frozen=True)
class RoundSpec:
    label: str
    training_steps: int
    renders: tuple[RenderSpec, ...]
    dataset_note: str
    result_note: str


RAINDEER_ROUNDS = (
    RoundSpec(
        label="round-1",
        training_steps=500,
        renders=(
            RenderSpec("raindeer-round-1-01-t2v", "t2v"),
            RenderSpec("raindeer-round-1-02-t2v", "t2v"),
            RenderSpec("raindeer-round-1-03-i2v", "i2v"),
        ),
        dataset_note="single tutorial-style character dataset",
        result_note="strongest photorealistic round; became the reference look",
    ),
    RoundSpec(
        label="round-2",
        training_steps=500,
        renders=(
            RenderSpec("raindeer-round-2-01-t2v", "t2v"),
            RenderSpec("raindeer-round-2-02-t2v", "t2v"),
            RenderSpec("raindeer-round-2-03-i2v", "i2v"),
        ),
        dataset_note="new reference footage with more visual variation",
        result_note="more synthetic; taught us to avoid mixed settings/outfits",
    ),
    RoundSpec(
        label="round-3",
        training_steps=1000,
        renders=(
            RenderSpec("raindeer-round-3-01-t2v", "t2v"),
            RenderSpec("raindeer-round-3-02-t2v", "t2v"),
            RenderSpec("raindeer-round-3-03-i2v", "i2v"),
        ),
        dataset_note="corridor-only subset, same outfit and lighting family",
        result_note="cleaner controlled comparison after excluding mixed-room clips",
    ),
)


def money(value: Decimal | str | float | int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_UP)


def training_cost(steps: int) -> Decimal:
    if steps < 100:
        raise ValueError("fal bills at least 100 training steps")
    return money(TRAINING_RATE_PER_STEP * steps)


def generated_megapixels(*, width: int, height: int, frames: int) -> Decimal:
    if min(width, height, frames) <= 0:
        raise ValueError("width, height, and frames must be positive")
    return Decimal(width * height * frames) / Decimal(1_000_000)


def quality_render_cost(*, width: int, height: int, frames: int) -> Decimal:
    return money(generated_megapixels(width=width, height=height, frames=frames) * QUALITY_RATE_PER_MEGAPIXEL)


def render_cost(render: RenderSpec) -> Decimal:
    return quality_render_cost(width=render.width, height=render.height, frames=render.frames)


def round_cost(round_spec: RoundSpec) -> Decimal:
    total = training_cost(round_spec.training_steps)
    for render in round_spec.renders:
        total += render_cost(render)
    return money(total)


def proof_file(path: Path, *, quality_status: str, approval_date: str = "2026-07-24") -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "quality_status": quality_status,
        "approval_date": approval_date,
        "manual_review": {
            "output_classification": "generated_output",
            "consent_or_authorization": "confirmed",
            "embedded_source_asset_metadata": False,
        },
    }


def proof_files(paths: list[Path], *, quality_status: str, approval_date: str = "2026-07-24") -> list[dict[str, Any]]:
    return [proof_file(path, quality_status=quality_status, approval_date=approval_date) for path in paths]


def rounds_as_dicts(rounds: tuple[RoundSpec, ...] = RAINDEER_ROUNDS) -> list[dict[str, Any]]:
    payload = []
    for round_spec in rounds:
        row = asdict(round_spec)
        row["training_cost_usd"] = str(training_cost(round_spec.training_steps))
        row["render_costs_usd"] = [str(render_cost(render)) for render in round_spec.renders]
        row["estimated_total_usd"] = str(round_cost(round_spec))
        payload.append(row)
    return payload


def write_round_plan(path: Path, rounds: tuple[RoundSpec, ...] = RAINDEER_ROUNDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "rounds": rounds_as_dicts(rounds)}, indent=2) + "\n")
