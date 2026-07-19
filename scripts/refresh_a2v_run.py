from __future__ import annotations

import argparse
from pathlib import Path

from ltx_lora_pilot.a2v_refresh import refresh_sealed_a2v_run
from ltx_lora_pilot.artifacts import canonical_json_bytes
from ltx_lora_pilot.private_workspace import approved_private_root_from_environment


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "A2V_REFRESH_ARGUMENT_ERROR\n")


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Issue a fresh immutable offline A2V run",
        allow_abbrev=False,
    )
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--source-execution-id", required=True)
    parser.add_argument("--expected-source-bundle-id", required=True)
    parser.add_argument("--target-execution-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    parser.add_argument("--price-evidence", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--validation-prompts", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    try:
        result = refresh_sealed_a2v_run(
            private_root=approved_private_root_from_environment(),
            pilot_id=args.pilot_id,
            source_execution_id=args.source_execution_id,
            expected_source_bundle_id=args.expected_source_bundle_id,
            target_execution_id=args.target_execution_id,
            created_at_utc=args.created_at_utc,
            expires_at_utc=args.expires_at_utc,
            fresh_price_evidence_path=args.price_evidence,
            fresh_standing_authorization_path=args.standing_authorization,
            validation_prompts_path=args.validation_prompts,
            repository_commit=args.repository_commit,
        )
    except Exception:
        parser.exit(2, "A2V_REFRESH_FAILED\n")

    print(
        canonical_json_bytes(
            {
                "status": "issued",
                "execution_id": result.execution_id,
                "bundle_id": result.bundle_id,
            }
        ).decode("utf-8")
    )


if __name__ == "__main__":
    main()
