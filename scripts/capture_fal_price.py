from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

from ltx_lora_pilot.artifacts import atomic_write_json
from ltx_lora_pilot.authorization import PriceEvidence, capture_price_evidence


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "FAL_PRICE_ARGUMENT_ERROR\n")


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _capture(
    output: Path,
    *,
    fetch: Callable[[str], bytes] | None = None,
    now: str | datetime | None = None,
) -> PriceEvidence:
    output_path = Path(output)
    if _is_symlink_or_junction(output_path) or output_path.exists():
        raise ValueError("price evidence destination already exists")
    if _is_symlink_or_junction(output_path.parent) or not output_path.parent.is_dir():
        raise ValueError("price evidence destination is unavailable")
    evidence = capture_price_evidence(fetch=fetch, now=now)
    atomic_write_json(output_path, evidence.to_dict())
    return evidence


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Capture unauthenticated official Fal A2V price evidence"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _capture(args.output)
    except Exception:
        parser.exit(2, "FAL_PRICE_CAPTURE_FAILED\n")
    print("FAL_PRICE_EVIDENCE_CAPTURED")


if __name__ == "__main__":
    main()
