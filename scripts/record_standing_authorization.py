from __future__ import annotations

import argparse
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ltx_lora_pilot.artifacts import canonical_json_bytes, sha256_file
from ltx_lora_pilot.authorization import (
    A2V_ENDPOINT,
    A2V_EXECUTIONS,
    A2V_STEPS,
    CUMULATIVE_CAP_USD,
    TRAINING_MAX_USD,
    VALIDATION_ALLOCATION_MAX_USD,
    StandingAuthorization,
)


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "STANDING_AUTHORIZATION_ARGUMENT_ERROR\n")


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_regular_file(path: Path) -> None:
    if _is_symlink_or_junction(path):
        raise ValueError("authorization source is unavailable")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError("authorization source is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("authorization source is unavailable")


def _require_new_output(path: Path) -> None:
    if _is_symlink_or_junction(path) or path.exists():
        raise ValueError("authorization destination already exists")
    if _is_symlink_or_junction(path.parent) or not path.parent.is_dir():
        raise ValueError("authorization destination is unavailable")


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
            raise ValueError("authorization destination already exists") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _record(
    source_file: Path,
    output: Path,
    *,
    policy_id: str,
    expires_at_utc: str,
    now: str | datetime | None = None,
) -> StandingAuthorization:
    source_path = Path(source_file)
    output_path = Path(output)
    _require_regular_file(source_path)
    _require_new_output(output_path)
    source_sha256 = sha256_file(source_path).sha256
    policy = StandingAuthorization.from_dict(
        {
            "policy_id": policy_id,
            "source_sha256": source_sha256,
            "endpoint": A2V_ENDPOINT,
            "executions": A2V_EXECUTIONS,
            "steps": A2V_STEPS,
            "training_max_usd": TRAINING_MAX_USD,
            "validation_allocation_usd": VALIDATION_ALLOCATION_MAX_USD,
            "cumulative_cap_usd": CUMULATIVE_CAP_USD,
            "expires_at_utc": expires_at_utc,
        },
        now=now,
    )
    _exclusive_atomic_write_json(output_path, policy.to_dict())
    return policy


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Record a private standing authorization by source hash"
    )
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _record(
            args.source_file,
            args.output,
            policy_id=args.policy_id,
            expires_at_utc=args.expires_at_utc,
        )
    except Exception:
        parser.exit(2, "STANDING_AUTHORIZATION_RECORD_FAILED\n")
    print("STANDING_AUTHORIZATION_RECORDED")


if __name__ == "__main__":
    main()
