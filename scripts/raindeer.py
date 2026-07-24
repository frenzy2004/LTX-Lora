from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltx_lora_pilot.raindeer import proof_files, rounds_as_dicts, write_round_plan  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Raindeer LTX character-LoRA workflow utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print the default round 1-3 cost plan")
    plan_parser.add_argument("--output", type=Path, help="optional JSON output path")

    proof_parser = subparsers.add_parser("proof", help="hash approved public proof videos")
    proof_parser.add_argument("videos", nargs="+", type=Path)
    proof_parser.add_argument("--output", type=Path, required=True)
    proof_parser.add_argument("--quality-status", default="raindeer_round_proof")
    proof_parser.add_argument("--approval-date", default="2026-07-24")

    args = parser.parse_args()
    if args.command == "plan":
        payload = {"schema_version": 1, "rounds": rounds_as_dicts()}
        if args.output:
            write_round_plan(args.output)
        print(json.dumps(payload, indent=2))
        return

    entries = proof_files(args.videos, quality_status=args.quality_status, approval_date=args.approval_date)
    _write_json(args.output, {"schema_version": 1, "files": entries})
    print(json.dumps({"manifest": str(args.output), "files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
