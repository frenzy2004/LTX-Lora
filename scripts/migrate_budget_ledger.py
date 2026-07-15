from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ltx_lora_pilot.pilot_ledger import migrate_legacy_ledger


class _NeutralParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ValueError("invalid migration arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _NeutralParser(description="Migrate the reviewed pilot budget ledger")
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        migrate_legacy_ledger(args.source_ledger, args.manifest, args.ledger)
    except Exception:
        print("error: ledger migration failed", file=sys.stderr)
        return 1
    print("migration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
