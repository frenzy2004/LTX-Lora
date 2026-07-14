from decimal import Decimal
from pathlib import Path

import pytest

from ltx_lora_pilot.budget import BudgetLedger
from ltx_lora_pilot.training import run_training


def test_upload_failure_releases_reservation(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")

    def fail_upload(_: Path) -> str:
        raise RuntimeError("upload failed before provider submission")

    with pytest.raises(RuntimeError, match="upload failed"):
        run_training(
            ledger=ledger,
            projected_cost=Decimal("1.20"),
            label="smoke",
            dataset=tmp_path / "dataset.zip",
            endpoint="test-endpoint",
            payload={},
            output=tmp_path / "result.json",
            upload_fn=fail_upload,
            submit_fn=lambda *_args, **_kwargs: {},
        )

    assert ledger.remaining() == Decimal("12.0000")


def test_submit_failure_keeps_conservative_charge(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")

    def fail_submit(*_args, **_kwargs):
        raise RuntimeError("provider submit status is uncertain")

    with pytest.raises(RuntimeError, match="status is uncertain"):
        run_training(
            ledger=ledger,
            projected_cost=Decimal("1.20"),
            label="smoke",
            dataset=tmp_path / "dataset.zip",
            endpoint="test-endpoint",
            payload={},
            output=tmp_path / "result.json",
            upload_fn=lambda _path: "https://private.invalid/dataset.zip",
            submit_fn=fail_submit,
        )

    assert ledger.remaining() == Decimal("10.8000")
