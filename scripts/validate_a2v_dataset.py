from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_lora_pilot.a2v_dataset import A2VSpec, validate_a2v_directory
from ltx_lora_pilot.a2v_quality import (
    load_quality_attestation,
    validate_quality_and_splits,
)
from ltx_lora_pilot.artifacts import atomic_write_json


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "A2V_ARGUMENT_ERROR\n")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_report_destination(
    *,
    dataset_dir: Path,
    attestation_path: Path,
    report_path: Path,
) -> None:
    dataset_root = dataset_dir.resolve(strict=False)
    destination = report_path.resolve(strict=False)
    if destination == dataset_root or dataset_root in destination.parents:
        raise ValueError("structural report destination is inside the dataset root")
    if _same_file(report_path, attestation_path):
        raise ValueError("structural report destination aliases the quality attestation")
    if report_path.exists() and dataset_root.is_dir():
        for candidate in dataset_root.iterdir():
            if _same_file(report_path, candidate):
                raise ValueError("structural report destination aliases a dataset entry")


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Validate an extracted fal LTX A2V dataset before upload"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--quality-attestation", type=Path, required=True)
    parser.add_argument("--structural-report", type=Path, required=True)
    parser.add_argument("--trigger-phrase")
    args = parser.parse_args()

    try:
        _validate_report_destination(
            dataset_dir=args.dataset_dir,
            attestation_path=args.quality_attestation,
            report_path=args.structural_report,
        )
        structural_report = validate_a2v_directory(
            args.dataset_dir,
            spec=A2VSpec(),
            trigger_phrase=args.trigger_phrase,
        )
        attestation = load_quality_attestation(args.quality_attestation)
        summary = validate_quality_and_splits(attestation, structural_report)
        atomic_write_json(args.structural_report, structural_report)
        output = json.dumps(summary, sort_keys=True)
    except Exception:
        parser.exit(2, "A2V_VALIDATION_FAILED\n")
    print(output)


if __name__ == "__main__":
    main()
