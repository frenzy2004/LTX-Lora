from __future__ import annotations

import argparse
from pathlib import Path

from ltx_lora_pilot.media import inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory private source videos with ffprobe")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = inventory(args.source)
    write_inventory(records, args.output)
    print(f"inventoried {len(records)} valid videos -> {args.output}")


if __name__ == "__main__":
    main()
