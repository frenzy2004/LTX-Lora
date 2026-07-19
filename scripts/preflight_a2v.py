from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from ltx_lora_pilot.artifacts import canonical_json_bytes
from ltx_lora_pilot.preflight import (
    PREFLIGHT_REPORT_SCHEMA,
    TRAINING_RESERVATION_USD,
    PreflightStatus,
    run_preflight,
)
from ltx_lora_pilot.private_workspace import approved_private_root_from_environment


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "A2V_PREFLIGHT_ARGUMENT_ERROR\n")


def _bundle_id(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid bundle ID")
    return value


def _root_failure(bundle_id: str, require_receipt: bool) -> PreflightStatus:
    return PreflightStatus(
        schema_version=PREFLIGHT_REPORT_SCHEMA,
        status="failed",
        failed_gate="private_root",
        receipt_required=require_receipt,
        bundle_id=bundle_id,
        execution_id=None,
        training_groups=None,
        holdout_groups=None,
        provider_validation_items=None,
        committed_usd=None,
        remaining_usd=None,
        training_reservation_usd=TRAINING_RESERVATION_USD,
        remaining_after_reservation_usd=None,
        passed_gates=(),
        pilot_id=None,
        ledger_id=None,
        ledger_head_sha256=None,
    )


def _print_report(report: PreflightStatus) -> None:
    print(canonical_json_bytes(report.to_public_dict()).decode("utf-8"))


def main() -> None:
    parser = _NeutralArgumentParser(description="Run an offline A2V bundle preflight")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--confirm-bundle-id", type=_bundle_id, required=True)
    parser.add_argument("--require-receipt", action="store_true")
    args = parser.parse_args()
    try:
        private_root = approved_private_root_from_environment()
    except Exception:
        report = _root_failure(args.confirm_bundle_id, args.require_receipt)
        _print_report(report)
        parser.exit(2)
    try:
        report = run_preflight(
            args.run_dir,
            args.confirm_bundle_id,
            require_receipt=args.require_receipt,
            approved_private_root=private_root,
            clock=lambda: datetime.now(timezone.utc).replace(microsecond=0),
        )
    except Exception:
        parser.exit(2, "A2V_PREFLIGHT_FAILED\n")
    _print_report(report)
    if report.status == "failed":
        parser.exit(2)


if __name__ == "__main__":
    main()
