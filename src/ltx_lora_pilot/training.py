from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ltx_lora_pilot.budget import BudgetLedger


UploadFunction = Callable[[Path], str]
SubmitFunction = Callable[..., Any]


def run_training(
    *,
    ledger: BudgetLedger,
    projected_cost: Decimal,
    label: str,
    dataset: Path,
    endpoint: str,
    payload: dict[str, Any],
    output: Path,
    upload_fn: UploadFunction,
    submit_fn: SubmitFunction,
    on_update: Callable[[Any], None] | None = None,
    on_enqueue: Callable[[str], None] | None = None,
) -> Any:
    reservation = ledger.reserve(projected_cost, label)
    reached_submit_boundary = False
    try:
        request_payload = dict(payload)
        request_payload["training_data_url"] = upload_fn(dataset)
        reached_submit_boundary = True
        result = submit_fn(endpoint, request_payload, on_update=on_update, on_enqueue=on_enqueue)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        ledger.finalize(reservation.id, consumed=reached_submit_boundary)
