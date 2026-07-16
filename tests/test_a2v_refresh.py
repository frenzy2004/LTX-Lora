from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from ltx_lora_pilot.a2v_bundle import (
    build_dataset_manifest,
    build_root_manifest,
    build_training_archive,
    compute_bundle_id,
)
from ltx_lora_pilot.a2v_quality import CHECK_KEYS, validate_quality_and_splits
from ltx_lora_pilot.a2v_refresh import (
    copy_accepted_candidates,
    verify_source_run_static,
)
from ltx_lora_pilot.artifacts import atomic_write_json, canonical_json_bytes, sha256_file
from ltx_lora_pilot.provider_validation import build_provider_validation_selection


PILOT_ID = "pilot_00000000000040008000000000000001"
EXECUTION_ID = "exec_00000000000040008000000000000002"
POLICY_ID = "policy_00000000000040008000000000000003"
LEDGER_ID = "ledger_00000000000040008000000000000004"


@dataclass(frozen=True)
class ReadySourceRun:
    private_root: Path
    pilot_id: str
    execution_id: str
    run_dir: Path
    bundle_id: str
    train_ids: list[str]
    holdout_ids: list[str]
    candidate_paths: tuple[Path, ...]


def _group_id(index: int) -> str:
    return f"grp_{index:012x}40008{index:015x}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _digest(path: Path) -> dict[str, Any]:
    return asdict(sha256_file(path))


def _policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "source_sha256": "1" * 64,
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "executions": 1,
        "steps": 1000,
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
        "expires_at_utc": "2026-07-14T12:00:00Z",
    }


def _price() -> dict[str, Any]:
    return {
        "source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "rate_usd_per_step": "0.006",
        "response_sha256": hashlib.sha256(b"expired static evidence").hexdigest(),
        "retrieved_at_utc": "2026-07-14T00:00:00Z",
        "expires_at_utc": "2026-07-14T12:00:00Z",
    }


def _make_candidates(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_dir.mkdir(parents=True)
    groups: list[dict[str, Any]] = []
    attested: list[dict[str, Any]] = []
    for index in range(1, 18):
        group_id = _group_id(index)
        paths: list[Path] = []
        for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
            path = candidate_dir / f"{group_id}{suffix}"
            path.write_bytes(f"synthetic-{index}-{suffix}".encode("ascii"))
            paths.append(path)
        groups.append(
            {
                "group_id": group_id,
                "files": [_digest(path) for path in sorted(paths, key=lambda item: item.name)],
            }
        )
        split = "train" if index <= 12 else "holdout"
        attested.append(
            {
                "group_id": group_id,
                "split": split,
                "accepted": True,
                "source_asset_id": f"asset_{index:02d}",
                "source_session_id": f"session_{index:02d}",
                "location_id": f"location_{index:02d}",
                "source_start_ms": 0,
                "source_end_ms": 3000,
                "checks": {key: True for key in CHECK_KEYS},
                "notes": "accepted synthetic fixture",
            }
        )
    return (
        {
            "schema_version": "a2v-structural-report-v1",
            "status": "valid",
            "spec": {
                "width": 544,
                "height": 960,
                "frames": 89,
                "fps": 24,
                "sample_rate": 48000,
            },
            "groups": groups,
        },
        {
            "schema_version": "a2v-quality-attestation-v1",
            "dataset_id": "dset_00000000000040008000000000000005",
            "rights_and_consent": {
                "confirmed": True,
                "reviewer_id": "reviewer_opaque_01",
                "reviewed_at_utc": "2026-07-14T00:10:00Z",
            },
            "groups": attested,
        },
    )


def _resolved_groups(
    structural: dict[str, Any], quality_summary: dict[str, Any], candidate_dir: Path
) -> list[dict[str, Any]]:
    train_ids = set(quality_summary["accepted_train_group_ids"])
    holdout_ids = set(quality_summary["accepted_holdout_group_ids"])
    result: list[dict[str, Any]] = []
    for group in structural["groups"]:
        group_id = group["group_id"]
        split = "train" if group_id in train_ids else "holdout" if group_id in holdout_ids else None
        if split is not None:
            result.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "files": [
                        {**record, "path": candidate_dir / record["name"]}
                        for record in group["files"]
                    ],
                }
            )
    return result


def _execution_config(
    *, policy: dict[str, Any], price: dict[str, Any], dataset_path: Path, archive_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": "a2v-execution-config-v2",
        "canonical_json_version": 1,
        "execution_id": EXECUTION_ID,
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "created_at_utc": "2026-07-14T00:20:00Z",
        "expires_at_utc": "2026-07-14T12:00:00Z",
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "trigger_phrase": "chrx9_speech",
        "rank": 32,
        "steps": 1000,
        "learning_rate": "0.0002",
        "training_frames": 89,
        "training_fps": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "audio_normalize": True,
        "audio_preserve_pitch": True,
        "debug_dataset": False,
        "negative_prompt": "synthetic artifacts, distortion",
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
        "dataset_manifest_sha256": sha256_file(dataset_path).sha256,
        "training_archive_sha256": sha256_file(archive_path).sha256,
        "standing_authorization_sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        "price_evidence_sha256": hashlib.sha256(canonical_json_bytes(price)).hexdigest(),
        "price_source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "rate_usd_per_step": "0.006",
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
    }


def _root_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "plan": sha256_file(run_dir / "plan.md"),
        "standing_authorization": sha256_file(run_dir / "control" / "standing-authorization.json"),
        "price_evidence": sha256_file(run_dir / "control" / "price-evidence.json"),
        "structural_report": sha256_file(run_dir / "control" / "structural-report.json"),
        "quality_attestation": sha256_file(run_dir / "control" / "quality-attestation.json"),
        "dataset_manifest": sha256_file(run_dir / "bundle" / "dataset-manifest.json"),
        "training_archive": sha256_file(run_dir / "bundle" / "training-data.zip"),
        "execution_config": sha256_file(run_dir / "control" / "execution-config.json"),
        "provider_validation_selection": sha256_file(
            run_dir / "validation" / "provider-validation-selection.json"
        ),
    }


def _secure_tree(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)


def _media_probe(path: Path, **_: Any) -> dict[str, Any]:
    if path.name.endswith("_end.mp4"):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 544,
                    "height": 960,
                    "avg_frame_rate": "24/1",
                    "r_frame_rate": "24/1",
                    "nb_read_frames": "89",
                    "start_time": "0",
                    "duration": "3.708333",
                    "time_base": "1/24",
                }
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "tags": {"major_brand": "isom"}},
            "frames": [
                {"media_type": "video", "best_effort_timestamp": str(index)}
                for index in range(89)
            ],
        }
    if path.name.endswith("_start.png"):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "png",
                    "pix_fmt": "rgb24",
                    "width": 544,
                    "height": 960,
                }
            ],
            "format": {"format_name": "png_pipe"},
        }
    if path.name.endswith("_audio.wav"):
        return {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_fmt": "s16",
                    "channels": 1,
                    "sample_rate": "48000",
                    "start_time": "0",
                    "duration": "3.708333",
                    "time_base": "1/48000",
                }
            ],
            "format": {"format_name": "wav"},
            "packets": [{"codec_type": "audio", "pts": "0"}],
        }
    raise AssertionError(f"unexpected media probe path: {path.name}")


def _first_frame_digest(path: Path, **_: Any) -> str:
    name = path.name
    group_id = name[: -len("_start.png")] if name.endswith("_start.png") else name[: -len("_end.mp4")]
    return hashlib.sha256(group_id.encode("ascii")).hexdigest()


def _tree_digests(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (sha256_file(path).bytes, sha256_file(path).sha256)
        for path in sorted(root.rglob("*"), key=lambda item: str(item))
        if path.is_file()
    }


def _expected_group_file_names(structural: dict[str, Any]) -> set[str]:
    return {
        record["name"]
        for group in structural["groups"]
        for record in group["files"]
    }


def _replace_with_link(path: Path) -> None:
    link_target = path.parent.parent / "linked-candidate-source.bin"
    link_target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(link_target)
    except OSError:
        os.link(link_target, path)


@pytest.fixture
def ready_source_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReadySourceRun:
    import ltx_lora_pilot.a2v_dataset as a2v_dataset
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(a2v_dataset, "_ffprobe", _media_probe)
    monkeypatch.setattr(a2v_dataset, "_first_frame_sha256", _first_frame_digest)
    monkeypatch.setattr(a2v_dataset, "_reject_digital_silence", lambda path: None)
    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)

    private_root = tmp_path / "private"
    run_dir = private_root / "pilots" / PILOT_ID / "runs" / EXECUTION_ID
    candidate_dir = run_dir / "candidates"
    control = run_dir / "control"
    bundle = run_dir / "bundle"
    validation = run_dir / "validation"
    for directory in (control, bundle, validation):
        directory.mkdir(parents=True, exist_ok=True)
    structural, quality = _make_candidates(candidate_dir)
    quality_summary = validate_quality_and_splits(quality, structural)
    archive_digest = build_training_archive(
        _resolved_groups(structural, quality_summary, candidate_dir),
        bundle / "training-data.zip",
    )
    dataset = build_dataset_manifest(structural, quality, archive_digest, candidate_dir=candidate_dir)
    _write_json(bundle / "dataset-manifest.json", dataset)
    policy = _policy()
    price = _price()
    _write_json(control / "standing-authorization.json", policy)
    _write_json(control / "price-evidence.json", price)
    _write_json(control / "structural-report.json", structural)
    _write_json(control / "quality-attestation.json", quality)
    config = _execution_config(
        policy=policy,
        price=price,
        dataset_path=bundle / "dataset-manifest.json",
        archive_path=bundle / "training-data.zip",
    )
    _write_json(control / "execution-config.json", config)
    selection = build_provider_validation_selection(
        structural_report=structural,
        quality_summary=quality_summary,
        execution_config=config,
        candidate_dir=candidate_dir,
        prompts={
            quality_summary["accepted_holdout_group_ids"][0]: "A close talking-head shot with natural speech.",
            quality_summary["accepted_holdout_group_ids"][1]: "A medium talking-head shot with steady eye contact.",
        },
    )
    _write_json(validation / "provider-validation-selection.json", selection)
    (run_dir / "plan.md").write_text("approved private synthetic plan", encoding="utf-8")
    root_manifest = build_root_manifest(
        execution_id=EXECUTION_ID,
        created_at_utc=config["created_at_utc"],
        expires_at_utc=config["expires_at_utc"],
        repository_commit="f" * 40,
        artifacts=_root_artifacts(run_dir),
        holdout_groups=dataset["groups"]["holdout"],
    )
    _write_json(bundle / "bundle-manifest.json", root_manifest)
    _secure_tree(private_root)
    return ReadySourceRun(
        private_root=private_root,
        pilot_id=PILOT_ID,
        execution_id=EXECUTION_ID,
        run_dir=run_dir,
        bundle_id=compute_bundle_id(root_manifest),
        train_ids=quality_summary["accepted_train_group_ids"],
        holdout_ids=quality_summary["accepted_holdout_group_ids"],
        candidate_paths=tuple(sorted(candidate_dir.iterdir(), key=lambda path: path.name)),
    )


def test_source_static_verifier_accepts_expired_execution_authority_when_bytes_are_bound(
    ready_source_run: ReadySourceRun,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )

    assert snapshot.run_dir == ready_source_run.run_dir
    assert snapshot.quality_summary["accepted_train_group_ids"] == ready_source_run.train_ids
    assert snapshot.quality_summary["accepted_holdout_group_ids"] == ready_source_run.holdout_ids


@pytest.mark.parametrize("mutation", ["wrong_bundle", "artifact_drift", "candidate_link"])
def test_source_static_verifier_rejects_unbound_or_aliased_source(
    ready_source_run: ReadySourceRun,
    mutation: str,
) -> None:
    expected_bundle_id = ready_source_run.bundle_id
    if mutation == "wrong_bundle":
        expected_bundle_id = "0" * 64
    elif mutation == "artifact_drift":
        (ready_source_run.run_dir / "control" / "structural-report.json").write_bytes(b"{}")
    else:
        _replace_with_link(ready_source_run.candidate_paths[0])

    with pytest.raises(ValueError):
        verify_source_run_static(
            private_root=ready_source_run.private_root,
            pilot_id=ready_source_run.pilot_id,
            source_execution_id=ready_source_run.execution_id,
            expected_source_bundle_id=expected_bundle_id,
        )


def test_source_static_verifier_never_mutates_the_source_run(
    ready_source_run: ReadySourceRun,
) -> None:
    before_tree = _tree_digests(ready_source_run.run_dir)
    before_mtime = ready_source_run.run_dir.stat().st_mtime_ns

    verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )

    assert _tree_digests(ready_source_run.run_dir) == before_tree
    assert ready_source_run.run_dir.stat().st_mtime_ns == before_mtime


def test_copy_accepted_candidates_uses_independent_regular_files(
    ready_source_run: ReadySourceRun,
    tmp_path: Path,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    source_before = _tree_digests(ready_source_run.run_dir)
    target = tmp_path / "staging" / "candidates"

    structural, attestation = copy_accepted_candidates(snapshot, target)

    assert len(structural["groups"]) == 17
    assert len(list(target.iterdir())) == 68
    assert all(path.stat().st_nlink == 1 for path in target.iterdir())
    assert {path.name for path in target.iterdir()} == _expected_group_file_names(structural)
    assert canonical_json_bytes(structural) == canonical_json_bytes(snapshot.structural_report)
    assert canonical_json_bytes(attestation) == canonical_json_bytes(snapshot.quality_attestation)
    assert _tree_digests(ready_source_run.run_dir) == source_before


def test_copy_accepted_candidates_rejects_a_destination_within_the_source_run(
    ready_source_run: ReadySourceRun,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    forbidden_target = ready_source_run.run_dir / "copy-destination"

    with pytest.raises(ValueError):
        copy_accepted_candidates(snapshot, forbidden_target)

    assert not forbidden_target.exists()


def test_copy_accepted_candidates_rejects_a_source_run_ancestor_alias(
    ready_source_run: ReadySourceRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    external_run = tmp_path / "external-source-run"
    ready_source_run.run_dir.rename(external_run)
    try:
        ready_source_run.run_dir.symlink_to(external_run, target_is_directory=True)
    except OSError:
        external_run.rename(ready_source_run.run_dir)
        import ltx_lora_pilot.a2v_refresh as a2v_refresh

        original = a2v_refresh._has_alias_component
        monkeypatch.setattr(
            a2v_refresh,
            "_has_alias_component",
            lambda path: Path(path) == snapshot.run_dir or original(Path(path)),
        )

    with pytest.raises(ValueError):
        copy_accepted_candidates(snapshot, tmp_path / "staging" / "candidates")


def test_copy_accepted_candidates_rejects_an_aliased_destination_without_writing(
    ready_source_run: ReadySourceRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    actual_parent = tmp_path / "actual-staging"
    actual_parent.mkdir()
    alias_parent = tmp_path / "alias-staging"
    try:
        alias_parent.symlink_to(actual_parent, target_is_directory=True)
    except OSError:
        original = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: Path(path) == alias_parent or original(path),
        )

    with pytest.raises(ValueError):
        copy_accepted_candidates(snapshot, alias_parent / "candidates")

    assert not (actual_parent / "candidates").exists()
