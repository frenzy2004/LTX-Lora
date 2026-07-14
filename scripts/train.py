from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from ltx_lora_pilot.budget import BudgetLedger, estimate_training_cost
from ltx_lora_pilot.fal_api import safe_console_text, submit, upload
from ltx_lora_pilot.training import run_training


ENDPOINT = "fal-ai/ltx23-trainer-v2/i2v"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit budget-capped LTX I2V LoRA training")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--budget", type=Decimal, default=Decimal("12.00"))
    parser.add_argument("--budget-state", type=Path, default=Path(".pilot_state/budget.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_result.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    projected = estimate_training_cost(args.steps)
    ledger = BudgetLedger(args.budget_state, cap_usd=args.budget)
    payload = {
        "training_data_url": "<uploaded-on-execute>",
        "trigger_phrase": "chrx9_person",
        "rank": 32,
        "number_of_steps": args.steps,
        "learning_rate": 0.0002,
        "first_frame_conditioning_p": 0.5,
        "number_of_frames": 89,
        "frame_rate": 24,
        "resolution": "medium",
        "aspect_ratio": "9:16",
        "auto_scale_input": True,
        "split_input_into_scenes": False,
        "with_audio": False,
    }
    print(json.dumps({"endpoint": ENDPOINT, "projected_cost_usd": str(projected), "remaining_budget_usd": str(ledger.remaining()), "payload": payload}, indent=2))
    if not args.execute:
        print("dry run only; pass --execute to upload and submit")
        return
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    request_state = args.output.with_suffix(".request.json")

    def record_request_id(request_id: str) -> None:
        request_state.parent.mkdir(parents=True, exist_ok=True)
        request_state.write_text(
            json.dumps({"endpoint": ENDPOINT, "request_id": request_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"request state written to {request_state}")

    run_training(
        ledger=ledger,
        projected_cost=projected,
        label=f"training:{args.steps}-steps",
        dataset=args.dataset,
        endpoint=ENDPOINT,
        payload=payload,
        output=args.output,
        upload_fn=upload,
        submit_fn=submit,
        on_update=lambda event: print(safe_console_text(event)),
        on_enqueue=record_request_id,
    )
    print(f"training result written to {args.output}")


if __name__ == "__main__":
    main()
