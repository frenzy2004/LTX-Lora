from decimal import Decimal

import pytest

from ltx_lora_pilot.budget import BudgetLedger, estimate_inference_cost, estimate_training_cost


def test_training_cost() -> None:
    assert estimate_training_cost(500) == Decimal("1.2000")
    assert estimate_training_cost(2000) == Decimal("4.8000")


def test_inference_cost() -> None:
    assert estimate_inference_cost("distilled", 1280, 720, 121) == Decimal("0.1567")


def test_ledger_blocks_overspend(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    first = ledger.reserve(Decimal("11.50"), "first")
    with pytest.raises(RuntimeError):
        ledger.reserve(Decimal("0.51"), "blocked")
    ledger.finalize(first.id, consumed=True)
    assert ledger.remaining() == Decimal("0.5000")


def test_released_reservation_restores_budget(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    reservation = ledger.reserve(Decimal("2.00"), "temporary")
    ledger.finalize(reservation.id, consumed=False)
    assert ledger.remaining() == Decimal("12.0000")
