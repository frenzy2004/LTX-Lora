from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_lora_pilot.a2v_execution import execute_training_bundle, system_utc_clock
from ltx_lora_pilot.preflight import run_preflight
from ltx_lora_pilot.private_workspace import approved_private_root_from_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the receipt-bound immutable LTX A2V training command"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--confirm-bundle-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="submit exactly one paid request after dry-run preflight succeeds",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        private_root = approved_private_root_from_environment()
        if not args.execute:
            report = run_preflight(
                args.run_dir,
                args.confirm_bundle_id,
                require_receipt=True,
                approved_private_root=private_root,
                clock=system_utc_clock,
            )
            print(json.dumps(report.to_public_dict(), sort_keys=True))
            return
        record = execute_training_bundle(
            args.run_dir,
            args.confirm_bundle_id,
            approved_private_root=private_root,
        )
    except Exception:
        parser.exit(2, "immutable A2V command failed; no automatic retry was performed\n")
    print(
        json.dumps(
            {
                "status": "submitted",
                "bundle_id": record.bundle_id,
                "execution_id": record.execution_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
