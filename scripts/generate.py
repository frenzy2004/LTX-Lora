from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from decimal import Decimal
from pathlib import Path

from ltx_lora_pilot.budget import BudgetLedger, estimate_inference_cost
from ltx_lora_pilot.fal_api import safe_console_text, submit, upload
from ltx_lora_pilot.generation import build_generation_request, resolve_uploaded_asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one budget-capped LTX generation")
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("t2v-lora", "t2v-base", "i2v-lora", "i2v-base", "a2v-lora"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora-file", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1248)
    parser.add_argument("--frames", type=int, default=89)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget", type=Decimal, default=Decimal("12.00"))
    parser.add_argument("--budget-state", type=Path, default=Path(".pilot_state/budget.json"))
    parser.add_argument("--asset-cache", type=Path, default=Path(".pilot_state/assets.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.name):
        raise ValueError("name must be a lowercase filesystem-safe slug")
    projected = estimate_inference_cost("distilled", args.width, args.height, args.frames)
    ledger = BudgetLedger(args.budget_state, cap_usd=args.budget)
    summary = {
        "name": args.name,
        "mode": args.mode,
        "projected_cost_usd": str(projected),
        "remaining_budget_usd": str(ledger.remaining()),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "execute": args.execute,
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print("dry run only; pass --execute to upload assets and submit")
        return

    lora_url = None
    image_url = None
    audio_url = None
    if args.mode.endswith("-lora"):
        if not args.lora_file:
            raise ValueError("LoRA mode requires --lora-file")
        lora_url = resolve_uploaded_asset(args.lora_file, args.asset_cache, upload)
    if args.mode.startswith("i2v") or args.image:
        if not args.image:
            raise ValueError("image-to-video mode requires --image")
        image_url = resolve_uploaded_asset(args.image, args.asset_cache, upload)
    if args.mode == "a2v-lora":
        if not args.audio:
            raise ValueError("audio-to-video mode requires --audio")
        audio_url = resolve_uploaded_asset(args.audio, args.asset_cache, upload)

    endpoint, payload = build_generation_request(
        mode=args.mode,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        lora_url=lora_url,
        image_url=image_url,
        audio_url=audio_url,
        lora_scale=args.lora_scale,
        seed=args.seed,
        generate_audio=args.generate_audio,
    )
    job_dir = args.output_dir / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    request_state = job_dir / "request.json"

    def record_request_id(request_id: str) -> None:
        request_state.write_text(
            json.dumps({"endpoint": endpoint, "request_id": request_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"request state written to {request_state}")

    reservation = ledger.reserve(projected, f"inference:{args.name}")
    reached_submit_boundary = False
    try:
        reached_submit_boundary = True
        result = submit(
            endpoint,
            payload,
            on_enqueue=record_request_id,
            on_update=lambda event: print(safe_console_text(event)),
        )
        result_path = job_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        video = result["video"]
        output_path = job_dir / "output.mp4"
        with urllib.request.urlopen(video["url"], timeout=120) as response, output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "charged_usd": str(projected),
                    "seed": result.get("seed"),
                    "duration": video.get("duration"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "fps": video.get("fps"),
                },
                indent=2,
            )
        )
    finally:
        ledger.finalize(reservation.id, consumed=reached_submit_boundary)


if __name__ == "__main__":
    main()
