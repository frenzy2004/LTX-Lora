from __future__ import annotations

import argparse
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ltx_lora_pilot.artifacts import (
    canonical_json_bytes,
    strict_load_json,
)
from ltx_lora_pilot.authorization import (
    APPROVAL_ID_PATTERN,
    EXECUTION_ID_PATTERN,
    PILOT_ID_PATTERN,
    PROCESS_ID_PATTERN,
    ExecutionReceipt,
    issue_execution_receipt,
)
from ltx_lora_pilot.pilot_ledger import LedgerPreflightSnapshot, PilotLedger
from ltx_lora_pilot.private_workspace import (
    approved_private_root_from_environment,
    require_canonical_run_dir,
    resolve_pilot_ledger,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "A2V_APPROVAL_ARGUMENT_ERROR\n")


def _bundle_id(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid bundle ID")
    return value


def _approval_id(value: str) -> str:
    if APPROVAL_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid approval ID")
    return value


def _issuer_process_id(value: str) -> str:
    if PROCESS_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid issuer process ID")
    return value


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _load_policy(path: Path) -> dict[str, object]:
    if _is_symlink_or_junction(path) or not path.is_file():
        raise ValueError("standing authorization is unavailable")
    value = strict_load_json(path)
    if type(value) is not dict or path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("standing authorization is not canonical")
    return value


def _lexical_run_identity(
    private_root: Path,
    run_dir: Path,
) -> tuple[str, str]:
    run_path = Path(run_dir)
    raw = str(run_path)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not run_path.is_absolute()
        or ".." in run_path.parts
    ):
        raise ValueError("canonical run directory is required")
    absolute = Path(os.path.abspath(run_path))
    if str(run_path) != str(absolute):
        raise ValueError("canonical run directory is required")
    root_parts = private_root.parts
    run_parts = run_path.parts
    if (
        len(run_parts) != len(root_parts) + 4
        or run_parts[: len(root_parts)] != root_parts
    ):
        raise ValueError("canonical run directory is required")
    relative = run_parts[len(root_parts) :]
    if relative[0] != "pilots" or relative[2] != "runs":
        raise ValueError("canonical run directory is required")
    pilot_id = relative[1]
    execution_id = relative[3]
    if (
        PILOT_ID_PATTERN.fullmatch(pilot_id) is None
        or EXECUTION_ID_PATTERN.fullmatch(execution_id) is None
    ):
        raise ValueError("canonical run directory is required")
    expected = private_root / "pilots" / pilot_id / "runs" / execution_id
    if str(expected) != str(run_path):
        raise ValueError("canonical run directory is required")
    return pilot_id, execution_id


def _exclusive_atomic_write_json(path: Path, value: Any) -> None:
    content = canonical_json_bytes(value)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError("execution approval already exists") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _issue(
    run_dir: Path,
    bundle_id: str,
    *,
    approval_id: str,
    issuer_process_id: str,
    now: str | datetime | None = None,
) -> ExecutionReceipt:
    private_root = approved_private_root_from_environment()
    pilot_id, execution_id = _lexical_run_identity(private_root, Path(run_dir))
    run_path = require_canonical_run_dir(
        private_root,
        pilot_id,
        execution_id,
        Path(run_dir),
    )
    ledger_path = resolve_pilot_ledger(private_root, pilot_id)
    control_dir = run_path / "control"
    approval_path = control_dir / "execution-approval.json"
    policy = _load_policy(control_dir / "standing-authorization.json")

    def read_ledger_snapshot(
        snapshot_pilot_id: str,
        ledger_id: str,
        snapshot_bundle_id: str,
        snapshot_execution_id: str,
    ) -> LedgerPreflightSnapshot:
        if (
            snapshot_pilot_id != pilot_id
            or snapshot_execution_id != execution_id
        ):
            raise ValueError("bundle identity does not match canonical run directory")
        ledger = PilotLedger.open_existing(
            ledger_path,
            pilot_id,
            expected_ledger_id=ledger_id,
        )
        return ledger.preflight_snapshot(
            snapshot_bundle_id,
            snapshot_execution_id,
        )

    receipt = issue_execution_receipt(
        policy,
        run_path,
        expected_bundle_id=bundle_id,
        read_ledger_snapshot=read_ledger_snapshot,
        approval_id=approval_id,
        issuer_process_id=issuer_process_id,
        now=now,
    )
    require_canonical_run_dir(
        private_root,
        pilot_id,
        execution_id,
        run_path,
    )
    _exclusive_atomic_write_json(approval_path, receipt.to_dict())
    return receipt


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Issue an offline approval for one immutable A2V bundle"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-id", type=_bundle_id, required=True)
    parser.add_argument("--approval-id", type=_approval_id, required=True)
    parser.add_argument(
        "--issuer-process-id",
        type=_issuer_process_id,
        required=True,
    )
    args = parser.parse_args()
    try:
        _issue(
            args.run_dir,
            args.bundle_id,
            approval_id=args.approval_id,
            issuer_process_id=args.issuer_process_id,
        )
    except Exception:
        parser.exit(2, "A2V_APPROVAL_ISSUE_FAILED\n")
    print("A2V_APPROVAL_ISSUED")


if __name__ == "__main__":
    main()
