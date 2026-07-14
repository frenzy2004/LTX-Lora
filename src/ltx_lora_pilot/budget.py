from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from pathlib import Path
from typing import Any


DEFAULT_CAP_USD = Decimal("12.00")
TRAINING_RATE_PER_STEP = Decimal("0.0024")
INFERENCE_RATES = {
    "distilled": Decimal("0.001405"),
    "full": Decimal("0.001805"),
    "quality": Decimal("0.0027075"),
}


def money(value: Decimal | str | float | int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_UP)


def estimate_training_cost(steps: int) -> Decimal:
    if steps < 100:
        raise ValueError("fal bills at least 100 training steps")
    return money(TRAINING_RATE_PER_STEP * steps)


def estimate_inference_cost(tier: str, width: int, height: int, frames: int) -> Decimal:
    if tier not in INFERENCE_RATES:
        raise ValueError(f"unknown tier: {tier}")
    if min(width, height, frames) <= 0:
        raise ValueError("dimensions and frames must be positive")
    megapixel_frames = Decimal(width * height * frames) / Decimal(1_000_000)
    return money(INFERENCE_RATES[tier] * megapixel_frames)


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: Decimal


class BudgetLedger:
    """Single-process atomic budget ledger for provider requests."""

    def __init__(self, path: Path, cap_usd: Decimal = DEFAULT_CAP_USD) -> None:
        self.path = path
        self.cap_usd = money(cap_usd)
        if self.cap_usd > DEFAULT_CAP_USD and os.getenv("ALLOW_BUDGET_OVERRIDE") != "1":
            raise ValueError("raising the default budget requires ALLOW_BUDGET_OVERRIDE=1")

    def _empty(self) -> dict[str, Any]:
        return {"cap_usd": str(self.cap_usd), "entries": []}

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        stored_cap = money(data["cap_usd"])
        if stored_cap != self.cap_usd:
            raise ValueError(f"ledger cap {stored_cap} does not match requested cap {self.cap_usd}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="budget-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def _committed(data: dict[str, Any]) -> Decimal:
        return sum(
            (money(entry["amount_usd"]) for entry in data["entries"] if entry["status"] in {"reserved", "consumed"}),
            Decimal("0"),
        )

    def remaining(self) -> Decimal:
        return money(self.cap_usd - self._committed(self.read()))

    def reserve(self, amount: Decimal, label: str) -> Reservation:
        amount = money(amount)
        data = self.read()
        remaining = self.cap_usd - self._committed(data)
        if amount <= 0:
            raise ValueError("reservation amount must be positive")
        if amount > remaining:
            raise RuntimeError(f"projected cost ${amount} exceeds remaining budget ${money(remaining)}")
        reservation = Reservation(id=str(uuid.uuid4()), amount=amount)
        data["entries"].append(
            {
                "id": reservation.id,
                "label": label,
                "amount_usd": str(amount),
                "status": "reserved",
                "created_at": int(time.time()),
            }
        )
        self._write(data)
        return reservation

    def finalize(self, reservation_id: str, consumed: bool) -> None:
        data = self.read()
        matches = [entry for entry in data["entries"] if entry["id"] == reservation_id]
        if len(matches) != 1:
            raise KeyError(f"unknown reservation: {reservation_id}")
        if matches[0]["status"] != "reserved":
            raise RuntimeError("reservation is already finalized")
        matches[0]["status"] = "consumed" if consumed else "released"
        matches[0]["finalized_at"] = int(time.time())
        self._write(data)
