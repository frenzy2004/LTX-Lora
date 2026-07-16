from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any
import zipfile

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
    refresh_sealed_a2v_run,
    verify_source_run_static,
)
from ltx_lora_pilot.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    strict_load_json,
)
from ltx_lora_pilot.authorization import (
    StandingAuthorization,
    capture_price_evidence,
    validate_execution_config,
)
from ltx_lora_pilot.pilot_ledger import migrate_legacy_ledger
from ltx_lora_pilot.preflight import run_preflight
from ltx_lora_pilot.provider_validation import (
    build_provider_validation_selection,
    validate_provider_validation_selection,
)


PILOT_ID = "pilot_00000000000040008000000000000001"
EXECUTION_ID = "exec_00000000000040008000000000000002"
POLICY_ID = "policy_00000000000040008000000000000003"
LEDGER_ID = "ledger_00000000000040008000000000000004"
FRESH_TARGET_EXECUTION_ID = "exec_00000000000040008000000000000006"
FRESH_CREATED_AT = "2026-07-16T01:00:00Z"
FRESH_EXPIRES_AT = "2026-07-16T12:00:00Z"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFRESH_SCRIPT = REPOSITORY_ROOT / "scripts" / "refresh_a2v_run.py"
REFRESH_MODULE = REFRESH_SCRIPT


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


@dataclass(frozen=True)
class FreshControls:
    target_execution_id: str
    created_at_utc: str
    expires_at_utc: str
    price_path: Path
    policy_path: Path
    prompts_path: Path
    prompts: dict[str, str]


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


def _fresh_policy() -> dict[str, Any]:
    value = _policy()
    value["expires_at_utc"] = FRESH_EXPIRES_AT
    return StandingAuthorization.from_dict(value, now=FRESH_CREATED_AT).to_dict()


def _fresh_price() -> dict[str, Any]:
    return capture_price_evidence(
        fetch=lambda _: (
            b"fal-ai/ltx23-trainer-v2/a2v Training costs 0.006 * steps; "
            b"1000 steps cost $6.00."
        ),
        now=FRESH_CREATED_AT,
    ).to_dict()


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


def _tree_metadata(root: Path) -> dict[str, tuple[int, int, int, int]]:
    paths = [root, *sorted(root.rglob("*"), key=lambda item: str(item))]
    return {
        str(path.relative_to(root)): (
            path.lstat().st_mode,
            path.lstat().st_nlink,
            path.lstat().st_size,
            path.lstat().st_mtime_ns,
        )
        for path in paths
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
    runs_dir = run_dir.parent
    runs_dir.mkdir(parents=True)
    if os.name == "nt":
        import ltx_lora_pilot.a2v_refresh as a2v_refresh

        # The issuer only creates children below an already-private runs root.
        a2v_refresh._normalize_windows_staging_dacl(runs_dir)
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


@pytest.fixture
def fresh_controls(ready_source_run: ReadySourceRun) -> FreshControls:
    controls = ready_source_run.private_root / "fresh-controls"
    price_path = controls / "price-evidence.json"
    policy_path = controls / "standing-authorization.json"
    prompts_path = controls / "validation-prompts.json"
    prompts = {
        ready_source_run.holdout_ids[0]: "A close talking-head shot with natural speech.",
        ready_source_run.holdout_ids[1]: "A medium talking-head shot with steady eye contact.",
    }
    _write_json(price_path, _fresh_price())
    _write_json(policy_path, _fresh_policy())
    _write_json(prompts_path, prompts)
    _secure_tree(ready_source_run.private_root)
    return FreshControls(
        target_execution_id=FRESH_TARGET_EXECUTION_ID,
        created_at_utc=FRESH_CREATED_AT,
        expires_at_utc=FRESH_EXPIRES_AT,
        price_path=price_path,
        policy_path=policy_path,
        prompts_path=prompts_path,
        prompts=prompts,
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL semantics")
@pytest.mark.parametrize("kind", ["directory", "file"])
def test_staging_privacy_rejects_explicit_nonowner_windows_ace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """The broad preflight guard must not be the sole Windows ACL gate."""

    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    candidate = tmp_path / f"named-ace-{kind}"
    if kind == "directory":
        candidate.mkdir()
        require_private = a2v_refresh._require_staging_directory
    else:
        candidate.write_bytes(b"private")
        require_private = a2v_refresh._require_staging_regular_file
    granted = subprocess.run(
        ["icacls", str(candidate), "/grant", "*S-1-5-32-545:(X)"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert granted.returncode == 0, granted.stdout + granted.stderr
    # Permit the existing broad check so this test specifically exercises the
    # required owner-SID comparison for the explicit BUILTIN\\Users ACE.
    monkeypatch.setattr(a2v_refresh._preflight, "_WINDOWS_DACL_CHECK", lambda _path: None)

    with pytest.raises(ValueError, match="not private"):
        require_private(candidate)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL semantics")
@pytest.mark.parametrize("kind", ["directory", "file"])
def test_windows_staging_dacl_rejects_inherited_interactive_ace_and_normalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    inherited_parent = tmp_path / "interactive-parent"
    inherited_parent.mkdir()
    granted = subprocess.run(
        ["icacls", str(inherited_parent), "/grant", "*S-1-5-4:(OI)(CI)(RX)"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert granted.returncode == 0, granted.stdout + granted.stderr
    candidate = inherited_parent / f"inherited-interactive-{kind}"
    if kind == "directory":
        candidate.mkdir()
        require_private = a2v_refresh._require_staging_directory
    else:
        candidate.write_bytes(b"private")
        require_private = a2v_refresh._require_staging_regular_file
    listed = subprocess.run(
        ["icacls", str(candidate)], capture_output=True, check=False, text=True
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "(I)" in listed.stdout
    assert "INTERACTIVE" in listed.stdout or "S-1-5-4" in listed.stdout

    monkeypatch.setattr(a2v_refresh._preflight, "_WINDOWS_DACL_CHECK", lambda _path: None)
    with pytest.raises(ValueError, match="not private"):
        require_private(candidate)

    with pytest.raises(ValueError, match="DACL"):
        a2v_refresh._require_windows_explicit_owner_only_dacl(candidate)

    a2v_refresh._normalize_windows_staging_dacl(candidate)
    a2v_refresh._require_windows_explicit_owner_only_dacl(candidate)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL semantics")
def test_refresh_refuses_an_interactive_runs_parent_before_staging(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    """A non-owner inheritable ACE on runs cannot race a new staging child."""

    runs = ready_source_run.run_dir.parent
    granted = subprocess.run(
        ["icacls", str(runs), "/grant", "*S-1-5-4:(OI)(CI)(M)"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert granted.returncode == 0, granted.stdout + granted.stderr
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError, match="not private"):
        refresh_sealed_a2v_run(**kwargs)

    assert not any(path.name.startswith(".a2v-refresh-") for path in runs.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL semantics")
def test_refresh_rejects_interactive_fresh_control_leaf_before_copy(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source control leaf never reaches copying with INTERACTIVE read access."""

    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    granted = subprocess.run(
        ["icacls", str(fresh_controls.price_path), "/grant", "*S-1-5-4:(RX)"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert granted.returncode == 0, granted.stdout + granted.stderr
    before = subprocess.run(
        ["icacls", str(fresh_controls.price_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert before.returncode == 0, before.stdout + before.stderr
    copied = False
    original_copy = a2v_refresh._copy_sealed_file

    def record_copy(*args: Any, **kwargs: Any) -> Any:
        nonlocal copied
        copied = True
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(a2v_refresh, "_copy_sealed_file", record_copy)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError, match="not private"):
        refresh_sealed_a2v_run(**kwargs)

    after = subprocess.run(
        ["icacls", str(fresh_controls.price_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert after.returncode == 0, after.stdout + after.stderr
    assert after.stdout == before.stdout
    assert not copied
    assert not any(
        path.name.startswith(".a2v-refresh-")
        for path in _target_run_dir(kwargs).parent.iterdir()
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL semantics")
def test_capture_fresh_control_rejects_inherited_interactive_leaf_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leaf check sees inherited ACEs before fresh controls are copied."""

    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    private_root = tmp_path / "private"
    inherited_parent = private_root / "controls"
    inherited_parent.mkdir(parents=True)
    granted = subprocess.run(
        ["icacls", str(inherited_parent), "/grant", "*S-1-5-4:(OI)(CI)(RX)"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert granted.returncode == 0, granted.stdout + granted.stderr
    leaf = inherited_parent / "price-evidence.json"
    leaf.write_text("{}", encoding="utf-8")
    before = subprocess.run(
        ["icacls", str(leaf)], capture_output=True, check=False, text=True
    )
    assert before.returncode == 0, before.stdout + before.stderr
    assert "(I)" in before.stdout
    assert "INTERACTIVE" in before.stdout or "S-1-5-4" in before.stdout
    monkeypatch.setattr(a2v_refresh._preflight, "_WINDOWS_DACL_CHECK", lambda _path: None)

    with pytest.raises(ValueError, match="not private"):
        a2v_refresh._capture_fresh_control(private_root, leaf)

    after = subprocess.run(
        ["icacls", str(leaf)], capture_output=True, check=False, text=True
    )
    assert after.returncode == 0, after.stdout + after.stderr
    assert after.stdout == before.stdout


def _refresh_kwargs(ready_source_run: ReadySourceRun, fresh_controls: FreshControls) -> dict[str, Any]:
    return {
        "private_root": ready_source_run.private_root,
        "pilot_id": ready_source_run.pilot_id,
        "source_execution_id": ready_source_run.execution_id,
        "expected_source_bundle_id": ready_source_run.bundle_id,
        "target_execution_id": fresh_controls.target_execution_id,
        "created_at_utc": fresh_controls.created_at_utc,
        "expires_at_utc": fresh_controls.expires_at_utc,
        "fresh_price_evidence_path": fresh_controls.price_path,
        "fresh_standing_authorization_path": fresh_controls.policy_path,
        "validation_prompts_path": fresh_controls.prompts_path,
        "repository_commit": "a" * 40,
    }


def _refresh_command(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> list[str]:
    return [
        sys.executable,
        str(REFRESH_SCRIPT),
        "--pilot-id",
        ready_source_run.pilot_id,
        "--source-execution-id",
        ready_source_run.execution_id,
        "--expected-source-bundle-id",
        ready_source_run.bundle_id,
        "--target-execution-id",
        fresh_controls.target_execution_id,
        "--created-at-utc",
        fresh_controls.created_at_utc,
        "--expires-at-utc",
        fresh_controls.expires_at_utc,
        "--price-evidence",
        str(fresh_controls.price_path),
        "--standing-authorization",
        str(fresh_controls.policy_path),
        "--validation-prompts",
        str(fresh_controls.prompts_path),
        "--repository-commit",
        "a" * 40,
    ]


def _refresh_environment(
    ready_source_run: ReadySourceRun,
    tmp_path: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["LTX_LORA_PRIVATE_ROOT"] = str(ready_source_run.private_root)
    if tmp_path is not None:
        shim_dir = tmp_path / "refresh-cli-shim"
        shim_dir.mkdir()
        (shim_dir / "sitecustomize.py").write_text(
            """
from hashlib import sha256
from pathlib import Path
import socket
import urllib.request

from ltx_lora_pilot import a2v_dataset, authorization, pilot_ledger, preflight


def _forbidden(*args, **kwargs):
    raise AssertionError("fresh issuer crossed an offline authority boundary")


def _media_probe(path, **_):
    name = Path(path).name
    if name.endswith("_end.mp4"):
        return {
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 544, "height": 960, "avg_frame_rate": "24/1", "r_frame_rate": "24/1", "nb_read_frames": "89", "start_time": "0", "duration": "3.708333", "time_base": "1/24"}],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "tags": {"major_brand": "isom"}},
            "frames": [{"media_type": "video", "best_effort_timestamp": str(index)} for index in range(89)],
        }
    if name.endswith("_start.png"):
        return {
            "streams": [{"codec_type": "video", "codec_name": "png", "pix_fmt": "rgb24", "width": 544, "height": 960}],
            "format": {"format_name": "png_pipe"},
        }
    if name.endswith("_audio.wav"):
        return {
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_fmt": "s16", "channels": 1, "sample_rate": "48000", "start_time": "0", "duration": "3.708333", "time_base": "1/48000"}],
            "format": {"format_name": "wav"},
            "packets": [{"codec_type": "audio", "pts": "0"}],
        }
    raise AssertionError("unexpected media probe path")


def _first_frame_digest(path, **_):
    name = Path(path).name
    group_id = name[:-len("_start.png")] if name.endswith("_start.png") else name[:-len("_end.mp4")]
    return sha256(group_id.encode("ascii")).hexdigest()


a2v_dataset._ffprobe = _media_probe
a2v_dataset._first_frame_sha256 = _first_frame_digest
a2v_dataset._reject_digital_silence = lambda path: None
preflight._WINDOWS_DACL_CHECK = lambda path: None
socket.socket = _forbidden
socket.create_connection = _forbidden
urllib.request.urlopen = _forbidden
authorization._fetch_official_price = _forbidden
authorization.issue_execution_receipt = _forbidden
authorization.verify_execution_receipt = _forbidden
pilot_ledger.PilotLedger.open_existing = _forbidden
pilot_ledger.PilotLedger.reserve = _forbidden
pilot_ledger.PilotLedger.reserve_training = _forbidden
""".lstrip(),
            encoding="utf-8",
        )
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(shim_dir)
            if not existing
            else os.pathsep.join((str(shim_dir), existing))
        )
    return environment


def _target_run_dir(kwargs: dict[str, Any]) -> Path:
    return (
        Path(kwargs["private_root"])
        / "pilots"
        / kwargs["pilot_id"]
        / "runs"
        / kwargs["target_execution_id"]
    )


def _load_root(run_dir: Path) -> dict[str, Any]:
    return strict_load_json(run_dir / "bundle" / "bundle-manifest.json")


def _target_split_counts(run_dir: Path) -> tuple[int, int]:
    dataset = strict_load_json(run_dir / "bundle" / "dataset-manifest.json")
    return dataset["counts"]["train_groups"], dataset["counts"]["holdout_groups"]


def _selected_holdout_ids(run_dir: Path) -> list[str]:
    selection = strict_load_json(run_dir / "validation" / "provider-validation-selection.json")
    return [item["group_id"] for item in selection["items"]]


def _archive_member_count(run_dir: Path) -> int:
    with zipfile.ZipFile(run_dir / "bundle" / "training-data.zip") as archive:
        return len(archive.infolist())


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


def test_copy_accepted_candidates_rejects_a_forged_external_snapshot_without_writing(
    ready_source_run: ReadySourceRun,
    tmp_path: Path,
) -> None:
    verified = verify_source_run_static(
        private_root=ready_source_run.private_root,
        pilot_id=ready_source_run.pilot_id,
        source_execution_id=ready_source_run.execution_id,
        expected_source_bundle_id=ready_source_run.bundle_id,
    )
    external_run = tmp_path / "unverified-external-run"
    external_candidates = external_run / "candidates"
    external_candidates.mkdir(parents=True)
    for source in ready_source_run.candidate_paths:
        (external_candidates / source.name).write_bytes(source.read_bytes())
    forged = replace(verified, run_dir=external_run)
    destination = tmp_path / "staging" / "candidates"

    with pytest.raises(ValueError):
        copy_accepted_candidates(forged, destination)

    assert not destination.exists()


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


def test_refresh_issues_a_fresh_bound_target_with_exact_split_and_selection(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    source_before = _tree_metadata(ready_source_run.run_dir)
    result = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert result.execution_id == fresh_controls.target_execution_id
    assert result.bundle_id == compute_bundle_id(_load_root(result.run_dir))
    assert _target_split_counts(result.run_dir) == (12, 5)
    assert _selected_holdout_ids(result.run_dir) == sorted(fresh_controls.prompts)
    assert _archive_member_count(result.run_dir) == 48
    dataset = strict_load_json(result.run_dir / "bundle" / "dataset-manifest.json")
    archive_names: set[str]
    with zipfile.ZipFile(result.run_dir / "bundle" / "training-data.zip") as archive:
        archive_names = {member.filename for member in archive.infolist()}
    assert archive_names == {record["name"] for record in dataset["training_members"]}
    assert archive_names.isdisjoint(
        {
            record["name"]
            for group in dataset["groups"]["holdout"]
            for record in group["files"]
        }
    )
    config = validate_execution_config(
        strict_load_json(result.run_dir / "control" / "execution-config.json")
    )
    source_config = strict_load_json(
        ready_source_run.run_dir / "control" / "execution-config.json"
    )
    assert config["execution_id"] == fresh_controls.target_execution_id
    assert (config["created_at_utc"], config["expires_at_utc"]) == (
        fresh_controls.created_at_utc,
        fresh_controls.expires_at_utc,
    )
    for field in ("trigger_phrase", "negative_prompt", "ledger_id"):
        assert config[field] == source_config[field]
    assert config["dataset_manifest_sha256"] == sha256_file(
        result.run_dir / "bundle" / "dataset-manifest.json"
    ).sha256
    assert config["training_archive_sha256"] == sha256_file(
        result.run_dir / "bundle" / "training-data.zip"
    ).sha256
    assert config["standing_authorization_sha256"] == sha256_file(
        result.run_dir / "control" / "standing-authorization.json"
    ).sha256
    assert config["price_evidence_sha256"] == sha256_file(
        result.run_dir / "control" / "price-evidence.json"
    ).sha256
    structural = strict_load_json(result.run_dir / "control" / "structural-report.json")
    attestation = strict_load_json(result.run_dir / "control" / "quality-attestation.json")
    selection = strict_load_json(
        result.run_dir / "validation" / "provider-validation-selection.json"
    )
    validate_provider_validation_selection(
        selection,
        structural,
        validate_quality_and_splits(attestation, structural),
        config,
        result.run_dir / "candidates",
    )
    assert set(_load_root(result.run_dir)["artifacts"]) == {
        "plan",
        "standing_authorization",
        "price_evidence",
        "structural_report",
        "quality_attestation",
        "dataset_manifest",
        "training_archive",
        "execution_config",
        "provider_validation_selection",
    }
    assert _tree_metadata(ready_source_run.run_dir) == source_before


def _configure_invalid_refresh_case(
    kwargs: dict[str, Any],
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    case: str,
) -> None:
    if case == "same_execution":
        kwargs["target_execution_id"] = ready_source_run.execution_id
    elif case == "existing_target":
        target = _target_run_dir(kwargs)
        target.mkdir(parents=True)
        (target / "sentinel").write_bytes(b"sentinel")
    elif case == "one_prompt":
        _write_json(fresh_controls.prompts_path, {ready_source_run.holdout_ids[0]: "One heldout prompt."})
    elif case == "three_prompts":
        _write_json(
            fresh_controls.prompts_path,
            {
                **fresh_controls.prompts,
                ready_source_run.holdout_ids[2]: "A third heldout prompt.",
            },
        )
    elif case == "train_prompt":
        _write_json(
            fresh_controls.prompts_path,
            {
                ready_source_run.train_ids[0]: "A training prompt must not be selected.",
                ready_source_run.holdout_ids[0]: "A heldout prompt remains valid.",
            },
        )
    elif case == "outside_private_root":
        outside = ready_source_run.private_root.parent / "outside-price-evidence.json"
        _write_json(outside, _fresh_price())
        kwargs["fresh_price_evidence_path"] = outside
    elif case == "linked_policy":
        _replace_with_link(fresh_controls.policy_path)
    elif case == "expired_price":
        value = _fresh_price()
        value["expires_at_utc"] = "2026-07-16T11:59:59Z"
        _write_json(fresh_controls.price_path, value)
    else:
        raise AssertionError(f"unexpected test case: {case}")


def _assert_no_new_target_or_preserved_sentinel(kwargs: dict[str, Any], case: str) -> None:
    target = _target_run_dir(kwargs)
    if case == "same_execution":
        assert target.is_dir()
    elif case == "existing_target":
        assert (target / "sentinel").read_bytes() == b"sentinel"
        assert list(target.iterdir()) == [target / "sentinel"]
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    "case",
    [
        "same_execution",
        "existing_target",
        "one_prompt",
        "three_prompts",
        "train_prompt",
        "outside_private_root",
        "linked_policy",
        "expired_price",
    ],
)
def test_refresh_rejects_untrusted_controls_or_target_contract_violations(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    case: str,
) -> None:
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)
    _configure_invalid_refresh_case(kwargs, ready_source_run, fresh_controls, case)
    source_before = _tree_digests(ready_source_run.run_dir)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert _tree_digests(ready_source_run.run_dir) == source_before
    _assert_no_new_target_or_preserved_sentinel(kwargs, case)


def test_refresh_never_overwrites_a_target_created_at_publication_time(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)
    target = _target_run_dir(kwargs)
    source_before = _tree_digests(ready_source_run.run_dir)

    def create_racing_sentinel(staging: Path, target_path: Path) -> bool:
        target_path.mkdir()
        (target_path / "sentinel").write_bytes(b"sentinel")
        return False

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", create_racing_sentinel)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert (target / "sentinel").read_bytes() == b"sentinel"
    assert _tree_digests(ready_source_run.run_dir) == source_before
    assert not [path for path in target.parent.iterdir() if path.name.startswith(".a2v-refresh-")]


def test_refresh_does_not_use_replace_for_final_publication(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    moved: list[tuple[Path, Path]] = []
    original = a2v_refresh._move_directory_no_replace

    def record_move(staging: Path, target: Path) -> bool:
        moved.append((staging, target))
        return original(staging, target)

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", record_move)

    result = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert result.run_dir.is_dir()
    assert moved == [(moved[0][0], result.run_dir)]
    publication_source = inspect.getsource(a2v_refresh._publish_new_run_no_replace)
    assert "os.replace" not in publication_source
    assert "os.rename" not in publication_source


@pytest.mark.skipif(os.name != "nt", reason="requires Windows MoveFileExW semantics")
def test_refresh_windows_no_replace_preserves_a_preexisting_target(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)
    target = _target_run_dir(kwargs)
    target.mkdir(parents=True)
    (target / "sentinel").write_bytes(b"sentinel")

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert (target / "sentinel").read_bytes() == b"sentinel"


def test_refresh_fails_closed_when_no_replace_primitive_is_unavailable(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", lambda staging, target: False)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert not _target_run_dir(_refresh_kwargs(ready_source_run, fresh_controls)).exists()


def test_refresh_refuses_before_staging_when_windows_runtime_is_unavailable(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    created_staging = False
    original_mkdtemp = a2v_refresh.tempfile.mkdtemp

    def record_mkdtemp(*args: Any, **kwargs: Any) -> str:
        nonlocal created_staging
        if kwargs.get("prefix") == a2v_refresh._STAGING_PREFIX:
            created_staging = True
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(
        a2v_refresh, "_windows_staging_runtime_available", lambda: False, raising=False
    )
    monkeypatch.setattr(a2v_refresh.tempfile, "mkdtemp", record_mkdtemp)
    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", lambda _s, _t: False)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError, match="Windows"):
        refresh_sealed_a2v_run(**kwargs)

    assert not created_staging
    assert not any(
        path.name.startswith(".a2v-refresh-")
        for path in _target_run_dir(kwargs).parent.iterdir()
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows runtime primitives")
@pytest.mark.parametrize(
    ("library_name", "primitive"),
    [
        ("Kernel32.dll", "SetFileInformationByHandle"),
        ("Advapi32.dll", "GetEffectiveRightsFromAclW"),
    ],
)
def test_refresh_refuses_before_staging_when_required_windows_primitive_is_missing(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
    library_name: str,
    primitive: str,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    created_staging = False
    source_verified = False
    original_mkdtemp = a2v_refresh.tempfile.mkdtemp
    original_windll = a2v_refresh.ctypes.WinDLL
    original_verify = a2v_refresh.verify_source_run_static

    class MissingPrimitiveLibrary:
        def __init__(self, library: Any) -> None:
            self._library = library

        def __getattr__(self, name: str) -> Any:
            if name == primitive:
                raise AttributeError(name)
            return getattr(self._library, name)

    def missing_primitive(name: str, *args: Any, **kwargs: Any) -> Any:
        library = original_windll(name, *args, **kwargs)
        if str(name).casefold() == library_name.casefold():
            return MissingPrimitiveLibrary(library)
        return library

    def record_source_verification(**kwargs: Any) -> Any:
        nonlocal source_verified
        source_verified = True
        return original_verify(**kwargs)

    def record_mkdtemp(*args: Any, **kwargs: Any) -> str:
        nonlocal created_staging
        if kwargs.get("prefix") == a2v_refresh._STAGING_PREFIX:
            created_staging = True
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(a2v_refresh.ctypes, "WinDLL", missing_primitive)
    monkeypatch.setattr(a2v_refresh.tempfile, "mkdtemp", record_mkdtemp)
    monkeypatch.setattr(a2v_refresh, "verify_source_run_static", record_source_verification)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError, match="Windows"):
        refresh_sealed_a2v_run(**kwargs)

    assert not created_staging
    assert not source_verified
    assert not any(
        path.name.startswith(".a2v-refresh-")
        for path in _target_run_dir(kwargs).parent.iterdir()
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows staging cleanup")
def test_refresh_rolls_back_empty_staging_when_dacl_normalization_fails(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    original_normalize = a2v_refresh._normalize_windows_staging_dacl

    def fail_for_new_staging(path: Path) -> None:
        if Path(path).name.startswith(a2v_refresh._STAGING_PREFIX):
            raise ValueError("injected DACL normalization failure")
        original_normalize(path)

    monkeypatch.setattr(
        a2v_refresh, "_normalize_windows_staging_dacl", fail_for_new_staging
    )
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError, match="private staging directory"):
        refresh_sealed_a2v_run(**kwargs)

    assert not any(
        path.name.startswith(".a2v-refresh-")
        for path in _target_run_dir(kwargs).parent.iterdir()
    )


def test_refresh_rechecks_staged_control_expiry_before_publication(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    original = a2v_refresh._fresh_execution_config

    def expire_staged_price(**kwargs: Any) -> dict[str, Any]:
        price_path = Path(kwargs["price_path"])
        value = strict_load_json(price_path)
        value["expires_at_utc"] = "2026-07-16T11:59:59Z"
        os.chmod(price_path, 0o600)
        _write_json(price_path, value)
        return original(**kwargs)

    monkeypatch.setattr(a2v_refresh, "_fresh_execution_config", expire_staged_price)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)
    source_before = _tree_digests(ready_source_run.run_dir)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert _tree_digests(ready_source_run.run_dir) == source_before
    assert not _target_run_dir(kwargs).exists()


def test_refresh_rejects_a_control_parent_replaced_after_canonicalization(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical input path cannot be reused through a swapped parent."""

    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    original_sha256 = a2v_refresh.sha256_file
    controls = fresh_controls.price_path.parent
    quarantined = controls.with_name("fresh-controls-original")
    swapped = False
    blocked = False

    def swap_parent_before_digest(path: Path) -> Any:
        nonlocal blocked, swapped
        if Path(path).name == fresh_controls.price_path.name and not swapped:
            try:
                controls.rename(quarantined)
            except OSError:
                blocked = True
                raise
            controls.mkdir()
            for name in (
                fresh_controls.price_path.name,
                fresh_controls.policy_path.name,
                fresh_controls.prompts_path.name,
            ):
                replacement = controls / name
                replacement.write_bytes((quarantined / name).read_bytes())
                if os.name != "nt":
                    os.chmod(replacement, 0o600)
            if os.name != "nt":
                os.chmod(controls, 0o700)
            swapped = True
        return original_sha256(path)

    monkeypatch.setattr(a2v_refresh, "sha256_file", swap_parent_before_digest)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert swapped or blocked
    assert not _target_run_dir(_refresh_kwargs(ready_source_run, fresh_controls)).exists()


def test_refresh_rechecks_staged_policy_ceiling_before_publication(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    original = a2v_refresh._fresh_execution_config

    def change_staged_policy_ceiling(**kwargs: Any) -> dict[str, Any]:
        policy_path = Path(kwargs["policy_path"])
        value = strict_load_json(policy_path)
        value["cumulative_cap_usd"] = "12.0001"
        os.chmod(policy_path, 0o600)
        _write_json(policy_path, value)
        return original(**kwargs)

    monkeypatch.setattr(
        a2v_refresh, "_fresh_execution_config", change_staged_policy_ceiling
    )
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert not _target_run_dir(kwargs).exists()


def test_refresh_rejects_a_published_tree_tampered_after_the_final_staging_check(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    def move_then_tamper(staging: Path, target: Path) -> bool:
        staging.rename(target)
        plan = target / "plan.md"
        os.chmod(plan, 0o600)
        plan.write_bytes(b"tampered after publication")
        if os.name != "nt":
            os.chmod(plan, 0o400)
        return True

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", move_then_tamper)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    assert (target := _target_run_dir(kwargs)).is_dir()
    assert (target / "plan.md").read_bytes() == b"tampered after publication"


def test_refresh_rejects_a_target_replaced_after_the_no_replace_move(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    def move_then_replace(staging: Path, target: Path) -> bool:
        staging.rename(target)
        target.rename(target.with_name("published-tree-quarantined"))
        target.mkdir()
        (target / "sentinel").write_bytes(b"replacement target")
        return True

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", move_then_replace)
    kwargs = _refresh_kwargs(ready_source_run, fresh_controls)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**kwargs)

    target = _target_run_dir(kwargs)
    assert (target / "sentinel").read_bytes() == b"replacement target"


def test_refresh_cleanup_fails_closed_on_a_nested_staging_alias_swap(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    external = ready_source_run.private_root.parent / "external-candidates"
    external.mkdir()
    protected_name = ready_source_run.candidate_paths[0].name
    protected = external / protected_name
    protected.write_bytes(b"external content must survive cleanup")
    original_walk = a2v_refresh._snapshot_staging_cleanup_tree
    armed = False
    swapped = False

    def move_failure(staging: Path, target: Path) -> bool:
        nonlocal armed
        armed = True
        return False

    def swap_after_walk(
        root: Path,
    ) -> tuple[list[Path], list[Path], list[Any]]:
        nonlocal swapped
        files, directories, records = original_walk(root)
        if armed and not swapped:
            candidates = Path(root) / "candidates"
            candidates.rename(Path(root).parent / ".quarantined-candidates")
            try:
                candidates.symlink_to(external, target_is_directory=True)
            except OSError:
                pytest.skip("directory symlinks are unavailable on this Windows host")
            swapped = True
        return files, directories, records

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", move_failure)
    monkeypatch.setattr(a2v_refresh, "_snapshot_staging_cleanup_tree", swap_after_walk)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert swapped
    assert protected.read_bytes() == b"external content must survive cleanup"


def test_refresh_cleanup_fails_closed_when_a_nested_directory_is_replaced(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup must not delete a normal directory swapped in after its walk."""

    import ltx_lora_pilot.a2v_refresh as a2v_refresh

    replacement = ready_source_run.private_root.parent / "attacker-candidates"
    replacement.mkdir()
    candidate_names = sorted(path.name for path in ready_source_run.candidate_paths)
    for name in candidate_names:
        (replacement / name).write_bytes(b"must survive cleanup")
        if os.name != "nt":
            os.chmod(replacement / name, 0o600)
    if os.name != "nt":
        os.chmod(replacement, 0o700)
    protected_name = candidate_names[0]
    original_walk = a2v_refresh._snapshot_staging_cleanup_tree
    armed = False
    swapped = False
    protected: Path | None = None

    def move_failure(staging: Path, target: Path) -> bool:
        nonlocal armed
        armed = True
        return False

    def swap_after_walk(
        root: Path,
    ) -> tuple[list[Path], list[Path], list[Any]]:
        nonlocal swapped, protected
        files, directories, records = original_walk(root)
        if armed and not swapped:
            candidates = Path(root) / "candidates"
            candidates.rename(Path(root).parent / ".quarantined-candidates")
            replacement.rename(candidates)
            protected = candidates / protected_name
            swapped = True
        return files, directories, records

    monkeypatch.setattr(a2v_refresh, "_move_directory_no_replace", move_failure)
    monkeypatch.setattr(a2v_refresh, "_snapshot_staging_cleanup_tree", swap_after_walk)

    with pytest.raises(ValueError):
        refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert swapped
    assert protected is not None
    assert protected.read_bytes() == b"must survive cleanup"


def _typed_legacy_id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:012x}40008{number:015x}"


def _install_matching_pilot_ledger(private_root: Path) -> None:
    amounts = ["1.2000", "0.1099", "0.1099", "0.3272", "0.3272", "1.4667"]
    states = ["consumed", "consumed", "consumed", "consumed", "reserved", "consumed"]
    source_entries: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for index, (amount, state) in enumerate(zip(amounts, states, strict=True), start=1):
        source_id = f"00000000-0000-4000-8000-{index:012x}"
        source_entries.append(
            {
                "id": source_id,
                "label": f"synthetic legacy item {index}",
                "amount_usd": amount,
                "status": state,
                "created_at": 1_700_000_000 + index,
                **(
                    {"finalized_at": 1_700_000_100 + index}
                    if state != "reserved"
                    else {}
                ),
            }
        )
        manifest_entries.append(
            {
                "source_entry_id": source_id,
                "reservation_id": _typed_legacy_id("reservation", index + 20),
                "bundle_id": hashlib.sha256(f"legacy-{index}".encode("ascii")).hexdigest(),
                "execution_id": _typed_legacy_id("exec", index + 40),
                "amount_usd": amount,
                "state": state,
            }
        )
    source = {"cap_usd": "12.0000", "entries": source_entries}
    manifest = {
        "schema_version": "pilot-budget-migration-v1",
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "migration_id": _typed_legacy_id("migration", 7),
        "cap_usd": "12.0000",
        "source_ledger_sha256": hashlib.sha256(canonical_json_bytes(source)).hexdigest(),
        "created_at_utc": "2026-07-16T00:00:00Z",
        "entries": manifest_entries,
    }
    ledger_dir = private_root / "pilots" / PILOT_ID / "ledger"
    evidence_dir = private_root / "migration-evidence"
    ledger_dir.mkdir(parents=True)
    evidence_dir.mkdir()
    source_path = evidence_dir / "legacy.json"
    manifest_path = evidence_dir / "manifest.json"
    _write_json(source_path, source)
    _write_json(manifest_path, manifest)
    migrate_legacy_ledger(source_path, manifest_path, ledger_dir / "pilot.sqlite3")
    _secure_tree(private_root)


def test_refresh_cli_issues_target_and_exposes_no_paid_or_provider_options(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        _refresh_command(ready_source_run, fresh_controls),
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=_refresh_environment(ready_source_run, tmp_path),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

    expected = {
        "status": "issued",
        "execution_id": fresh_controls.target_execution_id,
        "bundle_id": compute_bundle_id(
            _load_root(
                ready_source_run.private_root
                / "pilots"
                / ready_source_run.pilot_id
                / "runs"
                / fresh_controls.target_execution_id
            )
        ),
    }
    assert completed.stderr == ""
    assert completed.stdout == canonical_json_bytes(expected).decode("utf-8") + "\n"
    assert json.loads(completed.stdout) == expected
    assert "fal" not in completed.stdout.lower()

    help_result = subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT), "--help"],
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert help_result.stderr == ""
    help_text = help_result.stdout
    for forbidden in (
        "--private-root",
        "--fal-key",
        "--endpoint",
        "--budget",
        "--execute",
        "--submit",
        "--media-url",
    ):
        assert forbidden not in help_text


def test_refresh_cli_emits_neutral_argument_and_issuer_errors(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    malformed = subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT), "--unexpected", "private-value"],
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    assert malformed.returncode == 2
    assert malformed.stdout == ""
    assert malformed.stderr == "A2V_REFRESH_ARGUMENT_ERROR\n"

    failed = subprocess.run(
        _refresh_command(
            ready_source_run,
            replace(fresh_controls, target_execution_id=ready_source_run.execution_id),
        ),
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=_refresh_environment(ready_source_run),
        text=True,
    )
    assert failed.returncode == 2
    assert failed.stdout == ""
    assert failed.stderr == "A2V_REFRESH_FAILED\n"


def test_refresh_cli_rejects_abbreviated_and_forbidden_options(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    tmp_path: Path,
) -> None:
    environment = _refresh_environment(ready_source_run, tmp_path)
    abbreviated = _refresh_command(ready_source_run, fresh_controls)
    abbreviated[abbreviated.index("--pilot-id")] = "--pilot"
    completed = subprocess.run(
        abbreviated,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "A2V_REFRESH_ARGUMENT_ERROR\n"

    for forbidden in (
        "--private-root",
        "--fal-key",
        "--endpoint",
        "--budget",
        "--execute",
        "--submit",
        "--media-url",
        "--credential",
        "--api-key",
        "--receipt",
        "--ledger",
    ):
        rejected = subprocess.run(
            [*_refresh_command(ready_source_run, fresh_controls), forbidden, "private-value"],
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
        )
        assert rejected.returncode == 2
        assert rejected.stdout == ""
        assert rejected.stderr == "A2V_REFRESH_ARGUMENT_ERROR\n"


def test_fresh_issued_target_passes_policy_only_preflight(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    result = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))
    _install_matching_pilot_ledger(ready_source_run.private_root)

    status = run_preflight(
        result.run_dir,
        result.bundle_id,
        require_receipt=False,
        approved_private_root=ready_source_run.private_root,
        clock=lambda: datetime(2026, 7, 16, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert status.status == "ready_for_policy_issuance"
    assert status.failed_gate is None
    assert (status.training_groups, status.holdout_groups, status.provider_validation_items) == (
        12,
        5,
        2,
    )


def test_refresh_target_mutation_after_issuance_is_rejected_by_downstream_preflight(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
) -> None:
    result = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))
    _install_matching_pilot_ledger(ready_source_run.private_root)
    plan = result.run_dir / "plan.md"
    os.chmod(plan, 0o600)
    plan.write_bytes(b"mutated after issuance")
    if os.name != "nt":
        os.chmod(plan, 0o400)
    assert sha256_file(plan).sha256 != _load_root(result.run_dir)["artifacts"]["plan"]["sha256"]

    status = run_preflight(
        result.run_dir,
        result.bundle_id,
        require_receipt=False,
        approved_private_root=ready_source_run.private_root,
        clock=lambda: datetime(2026, 7, 16, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert status.status == "failed"
    assert status.failed_gate == "root_artifact_hashes"


def test_refresh_output_is_deterministic_for_identical_private_inputs(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    tmp_path: Path,
) -> None:
    second_private_root = tmp_path / "second-private"
    if os.name == "nt":
        import ltx_lora_pilot.a2v_refresh as a2v_refresh

        second_runs = (
            second_private_root
            / "pilots"
            / ready_source_run.pilot_id
            / "runs"
        )
        second_runs.mkdir(parents=True)
        a2v_refresh._normalize_windows_staging_dacl(second_runs)
        shutil.copytree(
            ready_source_run.run_dir,
            second_runs / ready_source_run.execution_id,
        )
        shutil.copytree(
            fresh_controls.price_path.parent,
            second_private_root
            / fresh_controls.price_path.parent.relative_to(ready_source_run.private_root),
        )
    else:
        shutil.copytree(ready_source_run.private_root, second_private_root)
    second_source = ReadySourceRun(
        private_root=second_private_root,
        pilot_id=ready_source_run.pilot_id,
        execution_id=ready_source_run.execution_id,
        run_dir=(
            second_private_root
            / "pilots"
            / ready_source_run.pilot_id
            / "runs"
            / ready_source_run.execution_id
        ),
        bundle_id=ready_source_run.bundle_id,
        train_ids=ready_source_run.train_ids,
        holdout_ids=ready_source_run.holdout_ids,
        candidate_paths=tuple(
            sorted(
                (
                    second_private_root
                    / "pilots"
                    / ready_source_run.pilot_id
                    / "runs"
                    / ready_source_run.execution_id
                    / "candidates"
                ).iterdir(),
                key=lambda path: path.name,
            )
        ),
    )
    second_controls = FreshControls(
        target_execution_id=fresh_controls.target_execution_id,
        created_at_utc=fresh_controls.created_at_utc,
        expires_at_utc=fresh_controls.expires_at_utc,
        price_path=second_private_root / fresh_controls.price_path.relative_to(ready_source_run.private_root),
        policy_path=second_private_root / fresh_controls.policy_path.relative_to(ready_source_run.private_root),
        prompts_path=second_private_root / fresh_controls.prompts_path.relative_to(ready_source_run.private_root),
        prompts=fresh_controls.prompts,
    )

    first = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))
    second = refresh_sealed_a2v_run(**_refresh_kwargs(second_source, second_controls))

    assert first.bundle_id == second.bundle_id
    assert _tree_digests(first.run_dir) == _tree_digests(second.run_dir)


def _assert_offline_only(source: str) -> None:
    for forbidden in (
        "fal_api",
        "a2v_execution",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "os.environ",
    ):
        assert forbidden not in source


def test_refresh_module_is_offline_only() -> None:
    _assert_offline_only(REFRESH_MODULE.read_text(encoding="utf-8"))


def test_refresh_issuer_module_is_offline_only() -> None:
    _assert_offline_only(
        (REPOSITORY_ROOT / "src" / "ltx_lora_pilot" / "a2v_refresh.py").read_text(
            encoding="utf-8"
        )
    )


def test_refresh_does_not_call_network_receipt_or_ledger_paths(
    ready_source_run: ReadySourceRun,
    fresh_controls: FreshControls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import urllib.request

    import ltx_lora_pilot.authorization as authorization
    import ltx_lora_pilot.pilot_ledger as pilot_ledger

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("fresh issuer crossed an offline authority boundary")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(authorization, "_fetch_official_price", forbidden)
    monkeypatch.setattr(authorization, "issue_execution_receipt", forbidden)
    monkeypatch.setattr(authorization, "verify_execution_receipt", forbidden)
    monkeypatch.setattr(pilot_ledger.PilotLedger, "open_existing", forbidden)
    monkeypatch.setattr(pilot_ledger.PilotLedger, "reserve", forbidden)
    monkeypatch.setattr(pilot_ledger.PilotLedger, "reserve_training", forbidden)

    result = refresh_sealed_a2v_run(**_refresh_kwargs(ready_source_run, fresh_controls))

    assert result.run_dir.is_dir()
