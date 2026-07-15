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
    PROCESS_ID_PATTERN,
    ExecutionReceipt,
    issue_execution_receipt,
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
    run_path = Path(run_dir)
    control_dir = run_path / "control"
    approval_path = control_dir / "execution-approval.json"
    policy = _load_policy(control_dir / "standing-authorization.json")
    receipt = issue_execution_receipt(
        policy,
        run_path,
        expected_bundle_id=bundle_id,
        approval_id=approval_id,
        issuer_process_id=issuer_process_id,
        now=now,
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
