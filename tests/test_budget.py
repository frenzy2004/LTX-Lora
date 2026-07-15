from decimal import Decimal

import pytest

import ltx_lora_pilot.budget as budget
from ltx_lora_pilot.budget import BudgetLedger, estimate_inference_cost, estimate_training_cost


def test_training_cost() -> None:
    assert estimate_training_cost(500) == Decimal("1.2000")
    assert estimate_training_cost(1_000, mode="a2v") == Decimal("6.0000")


def test_training_cost_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown training mode"):
        estimate_training_cost(1_000, mode="unknown")
    assert estimate_training_cost(2000) == Decimal("4.8000")


def test_inference_cost() -> None:
    assert estimate_inference_cost("distilled", 1280, 720, 121) == Decimal("0.1567")


def test_per_minute_cost_rounds_duration_up_to_whole_seconds() -> None:
    assert budget.estimate_per_minute_cost(Decimal("8"), Decimal("10.857333")) == Decimal("1.4667")
    assert budget.estimate_per_minute_cost(Decimal("8"), Decimal("60")) == Decimal("8.0000")


def test_json_budget_ledger_is_explicitly_legacy() -> None:
    assert "legacy" in (BudgetLedger.__doc__ or "").lower()


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
