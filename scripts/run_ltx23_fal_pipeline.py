from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

from ltx_lora_pilot.budget import BudgetLedger, DEFAULT_CAP_USD
from ltx_lora_pilot.fal_api import safe_console_text, submit, upload
from ltx_lora_pilot.generation import resolve_uploaded_asset
from ltx_lora_pilot.ltx23_v2 import (
    LTX23_T2V_TRAINER_ENDPOINT,
    build_ltx23_quality_lora_payload,
    build_ltx23_t2v_training_payload,
    estimate_ltx23_quality_inference_cost,
    estimate_ltx23_t2v_training_cost,
)


APPROVED_CAP_USD = Decimal("25.00")
PROMPTS = [
    'orvo says, "I think the real question is whether a sandwich can count as architecture."',
    'orvo says, "Coffee mugs have stronger opinions than most meetings, and I respect that."',
    (
        'orvo leans toward the camera and says, '
        '"Today I learned that every notebook secretly wants to become a weather report."'
    ),
]


def _make_ledger(path: Path, budget: Decimal) -> BudgetLedger:
    if DEFAULT_CAP_USD < budget <= APPROVED_CAP_USD:
        os.environ.setdefault("ALLOW_BUDGET_OVERRIDE", "1")
    return BudgetLedger(path, cap_usd=budget)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download_file(file_info: dict[str, Any], output_path: Path) -> None:
    url = file_info["url"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _record_request(path: Path, endpoint: str, request_id: str) -> None:
    _write_json(path, {"endpoint": endpoint, "request_id": request_id})
    print(f"request state written to {path}")


def _submit_training(
    *,
    dataset_zip: Path,
    ledger: BudgetLedger,
    asset_cache: Path,
    private_output_dir: Path,
) -> dict[str, Any]:
    projected = estimate_ltx23_t2v_training_cost(2000)
    reservation = ledger.reserve(projected, "training:ltx23-v2-t2v:2000-steps")
    reached_submit_boundary = False
    try:
        dataset_url = resolve_uploaded_asset(dataset_zip, asset_cache, upload)
        payload = build_ltx23_t2v_training_payload(training_data_url=dataset_url)
        _write_json(private_output_dir / "training_payload.json", payload)
        reached_submit_boundary = True
        result = submit(
            LTX23_T2V_TRAINER_ENDPOINT,
            payload,
            on_enqueue=lambda request_id: _record_request(
                private_output_dir / "training_request.json", LTX23_T2V_TRAINER_ENDPOINT, request_id
            ),
            on_update=lambda event: print(safe_console_text(event)),
        )
        _write_json(private_output_dir / "training_result.json", result)
        _download_file(result["lora_file"], private_output_dir / "orvo_lora.safetensors")
        _download_file(result["config_file"], private_output_dir / "orvo_lora_config.json")
        return result
    finally:
        ledger.finalize(reservation.id, consumed=reached_submit_boundary)


def _submit_generation(
    *,
    name: str,
    mode: str,
    prompt: str,
    lora_url: str,
    image_url: str | None,
    seed: int,
    ledger: BudgetLedger,
    private_output_dir: Path,
    public_output_dir: Path,
) -> dict[str, Any]:
    projected = estimate_ltx23_quality_inference_cost(5, "1080p")
    reservation = ledger.reserve(projected, f"inference:{name}")
    reached_submit_boundary = False
    endpoint, payload = build_ltx23_quality_lora_payload(
        mode=mode,
        prompt=prompt,
        lora_url=lora_url,
        image_url=image_url,
        seed=seed,
    )
    job_private_dir = private_output_dir / "generations" / name
    _write_json(job_private_dir / "payload.json", payload)
    try:
        reached_submit_boundary = True
        result = submit(
            endpoint,
            payload,
            on_enqueue=lambda request_id: _record_request(job_private_dir / "request.json", endpoint, request_id),
            on_update=lambda event: print(safe_console_text(event)),
        )
        _write_json(job_private_dir / "result.json", result)
        output_path = public_output_dir / f"{name}.mp4"
        _download_file(result["video"], output_path)
        return {
            "name": name,
            "mode": mode,
            "prompt": prompt,
            "seed": result.get("seed", seed),
            "endpoint": endpoint,
            "output": str(output_path),
            "result": result,
            "estimated_cost_usd": str(projected),
        }
    finally:
        ledger.finalize(reservation.id, consumed=reached_submit_boundary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the approved LTX-2.3/FAL character LoRA pipeline")
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--budget", type=Decimal, default=APPROVED_CAP_USD)
    parser.add_argument("--budget-state", type=Path, default=Path("private_work/ltx23_orvo/budget.json"))
    parser.add_argument("--asset-cache", type=Path, default=Path("private_work/ltx23_orvo/assets.json"))
    parser.add_argument("--private-output-dir", type=Path, default=Path("private_work/ltx23_orvo/fal"))
    parser.add_argument("--public-output-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ledger = _make_ledger(args.budget_state, args.budget)
    training_cost = estimate_ltx23_t2v_training_cost(2000)
    generation_cost = estimate_ltx23_quality_inference_cost(5, "1080p")
    planned = {
        "budget_usd": str(args.budget),
        "remaining_budget_usd": str(ledger.remaining()),
        "training_endpoint": LTX23_T2V_TRAINER_ENDPOINT,
        "training_cost_usd": str(training_cost),
        "generation_cost_each_usd": str(generation_cost),
        "prompts": PROMPTS,
        "execute": args.execute,
    }
    print(json.dumps(planned, indent=2))
    if args.dry_run or not args.execute:
        return
    if not args.dataset_zip.is_file():
        raise FileNotFoundError(args.dataset_zip)

    training_result = _submit_training(
        dataset_zip=args.dataset_zip,
        ledger=ledger,
        asset_cache=args.asset_cache,
        private_output_dir=args.private_output_dir,
    )
    lora_url = training_result["lora_file"]["url"]

    image_url = None
    if args.reference_image:
        image_url = resolve_uploaded_asset(args.reference_image, args.asset_cache, upload)

    generations = []
    for index, prompt in enumerate(PROMPTS, start=1):
        mode = "i2v" if image_url and index == len(PROMPTS) else "t2v"
        generations.append(
            _submit_generation(
                name=f"ltx23_orvo_{index:02d}_{mode}",
                mode=mode,
                prompt=prompt,
                lora_url=lora_url,
                image_url=image_url if mode == "i2v" else None,
                seed=4100 + index,
                ledger=ledger,
                private_output_dir=args.private_output_dir,
                public_output_dir=args.public_output_dir,
            )
        )

    manifest = {
        "training": {
            "endpoint": LTX23_T2V_TRAINER_ENDPOINT,
            "estimated_cost_usd": str(training_cost),
            "lora_file": str(args.private_output_dir / "orvo_lora.safetensors"),
            "config_file": str(args.private_output_dir / "orvo_lora_config.json"),
        },
        "generations": generations,
        "budget": ledger.read(),
    }
    _write_json(args.public_output_dir / "ltx23_orvo_manifest.json", manifest)
    print(json.dumps({"manifest": str(args.public_output_dir / "ltx23_orvo_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
