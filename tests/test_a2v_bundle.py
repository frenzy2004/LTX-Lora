from __future__ import annotations

import hashlib
import json
import stat
import struct
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ltx_lora_pilot.a2v_bundle import (
    build_dataset_manifest,
    build_root_manifest,
    build_training_archive,
    compute_bundle_id,
    inspect_training_archive,
)
from ltx_lora_pilot.a2v_quality import CHECK_KEYS
from ltx_lora_pilot.artifacts import FileDigest, canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_a2v_bundle.py"
FIXED_DATETIME = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR = (stat.S_IFREG | 0o600) << 16
REQUIRED_ROOT_ARTIFACTS = {
    "dataset_manifest",
    "execution_config",
    "plan",
    "price_evidence",
    "provider_validation_selection",
    "quality_attestation",
    "standing_authorization",
    "structural_report",
    "training_archive",
}


def _file_record(path: Path, name: str) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "name": name,
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _make_group(root: Path, group_id: str, split: str) -> dict[str, Any]:
    files = []
    for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
        name = f"{group_id}{suffix}"
        path = root / name
        path.write_bytes(f"{group_id}:{suffix}".encode("ascii"))
        files.append(_file_record(path, name))
    return {"group_id": group_id, "split": split, "files": files}


def _fixture_groups(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        _make_group(tmp_path, "sample_002", "train"),
        _make_group(tmp_path, "sample_003", "holdout"),
        _make_group(tmp_path, "sample_001", "train"),
    ]


def _digest_record(name: str, content: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_zip(
    path: Path,
    entries: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
    external_attr: int = FIXED_EXTERNAL_ATTR,
    extra: bytes = b"",
    member_comment: bytes = b"",
    archive_comment: bytes = b"",
) -> list[dict[str, Any]]:
    expected = []
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.comment = archive_comment
        for name, content in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_DATETIME)
            info.create_system = 3
            info.external_attr = external_attr
            info.compress_type = compression
            info.extra = extra
            info.comment = member_comment
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(info, content)
            expected.append(_digest_record(name, content))
    return expected


def _mark_zip_encrypted(path: Path) -> None:
    content = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        cursor = 0
        while True:
            index = content.find(signature, cursor)
            if index < 0:
                break
            flags = struct.unpack_from("<H", content, index + flag_offset)[0]
            struct.pack_into("<H", content, index + flag_offset, flags | 0x1)
            cursor = index + len(signature)
    path.write_bytes(content)


def _artifact_digest(label: str) -> FileDigest:
    content = f"artifact:{label}".encode("ascii")
    return FileDigest(
        name=f"{label}.json",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _reports(
    *,
    candidate_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    structural_groups = []
    quality_groups = []
    for index in range(15):
        group_id = f"sample_{index + 1:03d}"
        split = "train" if index < 10 else "holdout"
        files = []
        for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
            name = f"{group_id}{suffix}"
            content = f"{group_id}:{suffix}".encode("ascii")
            if candidate_root is not None:
                (candidate_root / name).write_bytes(content)
            files.append(_digest_record(name, content))
        structural_groups.append({"group_id": group_id, "files": files})
        quality_groups.append(
            {
                "group_id": group_id,
                "split": split,
                "accepted": True,
                "source_asset_id": f"asset-{index + 1}",
                "source_session_id": f"session-{index + 1}",
                "location_id": (
                    f"train-location-{index + 1}"
                    if split == "train"
                    else f"holdout-location-{index - 9}"
                ),
                "source_start_ms": index * 10_000,
                "source_end_ms": index * 10_000 + 3_708,
                "checks": {name: True for name in CHECK_KEYS},
                "notes": "",
            }
        )
    structural = {
        "schema_version": "a2v-structural-report-v1",
        "status": "valid",
        "spec": {
            "width": 544,
            "height": 960,
            "frames": 89,
            "fps": 24,
            "sample_rate": 48_000,
        },
        "groups": list(reversed(structural_groups)),
    }
    attestation = {
        "schema_version": "a2v-quality-attestation-v1",
        "dataset_id": "dataset-001",
        "rights_and_consent": {
            "confirmed": True,
            "reviewer_id": "reviewer-opaque-001",
            "reviewed_at_utc": "2026-07-15T00:00:00Z",
        },
        "groups": list(reversed(quality_groups)),
    }
    return structural, attestation


def test_archive_is_byte_identical_across_two_builds(tmp_path: Path) -> None:
    groups = _fixture_groups(tmp_path)

    first = build_training_archive(groups, tmp_path / "one.zip")
    second = build_training_archive(groups, tmp_path / "two.zip")

    assert first.sha256 == second.sha256
    assert first.bytes == second.bytes
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()


def test_archive_metadata_and_order_are_fully_deterministic(tmp_path: Path) -> None:
    groups = _fixture_groups(tmp_path)

    build_training_archive(groups, tmp_path / "training.zip")

    with zipfile.ZipFile(tmp_path / "training.zip") as archive:
        infos = archive.infolist()
        assert archive.comment == b""
        assert [info.filename for info in infos] == sorted(
            info.filename for info in infos
        )
        assert all(info.date_time == FIXED_DATETIME for info in infos)
        assert all(info.create_system == 3 for info in infos)
        assert all(info.external_attr == FIXED_EXTERNAL_ATTR for info in infos)
        assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
        assert all(info.extra == b"" and info.comment == b"" for info in infos)
        assert all(not (info.flag_bits & 0x1) for info in infos)


def test_archive_contains_train_groups_only(tmp_path: Path) -> None:
    groups = _fixture_groups(tmp_path)

    build_training_archive(groups, tmp_path / "training.zip")

    with zipfile.ZipFile(tmp_path / "training.zip") as archive:
        names = archive.namelist()
        assert len(names) == 8
        assert all(not name.startswith("sample_003") for name in names)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:drive-relative.txt",
        "nested/member.txt",
        "./dot.txt",
        "back\\slash.txt",
    ],
)
def test_inspection_rejects_non_root_or_unsafe_names(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    path = tmp_path / "unsafe.zip"
    _write_zip(path, [(unsafe_name, b"content")])

    with pytest.raises(ValueError, match="member name"):
        inspect_training_archive(path, [])


def test_inspection_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    _write_zip(path, [("sample.txt", b"one"), ("sample.txt", b"two")])
    expected = [_digest_record("sample.txt", b"one")]

    with pytest.raises(ValueError, match="duplicate"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_case_colliding_names(tmp_path: Path) -> None:
    path = tmp_path / "case-collision.zip"
    _write_zip(path, [("sample.txt", b"one"), ("SAMPLE.txt", b"two")])

    with pytest.raises(ValueError, match="case-colliding"):
        inspect_training_archive(path, [])


def test_inspection_rejects_symlink_attributes(tmp_path: Path) -> None:
    path = tmp_path / "symlink.zip"
    expected = _write_zip(
        path,
        [("sample.txt", b"target")],
        external_attr=(stat.S_IFLNK | 0o777) << 16,
    )

    with pytest.raises(ValueError, match="regular-file attributes"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_encrypted_members_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.zip"
    expected = _write_zip(path, [("sample.txt", b"not-really-encrypted")])
    _mark_zip_encrypted(path)

    with pytest.raises(ValueError, match="encryption"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_non_stored_members(tmp_path: Path) -> None:
    path = tmp_path / "compressed.zip"
    expected = _write_zip(
        path,
        [("sample.txt", bytes(range(256)))],
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(ValueError, match="ZIP_STORED"):
        inspect_training_archive(path, expected)


def test_inspection_enforces_member_count_limit(tmp_path: Path) -> None:
    path = tmp_path / "too-many.zip"
    expected = _write_zip(path, [("a.txt", b"a"), ("b.txt", b"b")])

    with pytest.raises(ValueError, match="member-count limit"):
        inspect_training_archive(path, expected, max_members=1)


def test_inspection_enforces_uncompressed_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "too-large.zip"
    expected = _write_zip(path, [("sample.txt", b"0123456789")])

    with pytest.raises(ValueError, match="uncompressed-size limit"):
        inspect_training_archive(path, expected, max_uncompressed_bytes=9)


def test_inspection_enforces_compression_ratio_limit(tmp_path: Path) -> None:
    path = tmp_path / "ratio.zip"
    expected = _write_zip(
        path,
        [("sample.txt", b"0" * 10_000)],
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(ValueError, match="compression-ratio limit"):
        inspect_training_archive(path, expected, max_compression_ratio=2)


@pytest.mark.parametrize(
    ("extra", "member_comment", "archive_comment"),
    [
        (b"\x01\x00\x00\x00", b"", b""),
        (b"", b"comment", b""),
        (b"", b"", b"comment"),
    ],
)
def test_inspection_rejects_comments_and_extra_fields(
    tmp_path: Path,
    extra: bytes,
    member_comment: bytes,
    archive_comment: bytes,
) -> None:
    path = tmp_path / "metadata.zip"
    expected = _write_zip(
        path,
        [("sample.txt", b"content")],
        extra=extra,
        member_comment=member_comment,
        archive_comment=archive_comment,
    )

    with pytest.raises(ValueError, match="metadata"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_unexpected_members(tmp_path: Path) -> None:
    path = tmp_path / "unexpected.zip"
    _write_zip(path, [("expected.txt", b"expected"), ("extra.txt", b"extra")])
    expected = [_digest_record("expected.txt", b"expected")]

    with pytest.raises(ValueError, match="unexpected members"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_missing_members(tmp_path: Path) -> None:
    path = tmp_path / "missing.zip"
    expected = _write_zip(path, [("present.txt", b"present")])
    expected.append(_digest_record("missing.txt", b"missing"))

    with pytest.raises(ValueError, match="missing members"):
        inspect_training_archive(path, expected)


def test_inspection_rejects_changed_member_bytes(tmp_path: Path) -> None:
    path = tmp_path / "changed.zip"
    _write_zip(path, [("sample.txt", b"changed")])
    expected = [_digest_record("sample.txt", b"original")]

    with pytest.raises(ValueError, match="does not match its manifest"):
        inspect_training_archive(path, expected)


def test_builder_rejects_source_changed_after_structural_validation(
    tmp_path: Path,
) -> None:
    groups = _fixture_groups(tmp_path)
    source = groups[0]["files"][0]["path"]
    source.write_bytes(b"changed-after-validation")

    with pytest.raises(ValueError, match="does not match its structural digest"):
        build_training_archive(groups, tmp_path / "training.zip")
    assert not (tmp_path / "training.zip").exists()


def test_dataset_manifest_binds_train_holdout_reports_and_archive() -> None:
    structural, attestation = _reports()
    archive = _artifact_digest("training-data")

    manifest = build_dataset_manifest(structural, attestation, archive)

    structural_bytes = canonical_json_bytes(structural)
    attestation_bytes = canonical_json_bytes(attestation)
    assert manifest["schema_version"] == "a2v-dataset-manifest-v1"
    assert manifest["dataset_id"] == "dataset-001"
    assert manifest["counts"] == {"holdout_groups": 5, "train_groups": 10}
    assert len(manifest["training_members"]) == 40
    assert len(manifest["groups"]["train"]) == 10
    assert len(manifest["groups"]["holdout"]) == 5
    assert manifest["reports"]["structural_report"] == {
        "bytes": len(structural_bytes),
        "sha256": hashlib.sha256(structural_bytes).hexdigest(),
    }
    assert manifest["reports"]["quality_attestation"] == {
        "bytes": len(attestation_bytes),
        "sha256": hashlib.sha256(attestation_bytes).hexdigest(),
    }
    assert manifest["archive"] == {
        "bytes": archive.bytes,
        "sha256": archive.sha256,
    }
    assert manifest["archive_builder_version"] == "a2v-bundle-builder-v1"
    assert [item["name"] for item in manifest["training_members"]] == sorted(
        item["name"] for item in manifest["training_members"]
    )
    serialized = canonical_json_bytes(manifest)
    assert b"source_asset_id" not in serialized
    assert b"source_session_id" not in serialized
    assert b"reviewer_id" not in serialized


def test_dataset_manifest_verifies_all_candidate_bytes(tmp_path: Path) -> None:
    structural, attestation = _reports(candidate_root=tmp_path)
    holdout_name = "sample_015_end.mp4"
    (tmp_path / holdout_name).write_bytes(b"changed-holdout")

    with pytest.raises(ValueError, match="does not match its structural digest"):
        build_dataset_manifest(
            structural,
            attestation,
            _artifact_digest("training-data"),
            candidate_dir=tmp_path,
        )


def test_dataset_manifest_rejects_free_form_identifying_dataset_id() -> None:
    structural, attestation = _reports()
    attestation["dataset_id"] = "Private Person Name"

    with pytest.raises(ValueError, match="canonical opaque ID"):
        build_dataset_manifest(
            structural,
            attestation,
            _artifact_digest("training-data"),
        )


def test_root_manifest_has_an_explicit_digest_domain() -> None:
    structural, attestation = _reports()
    dataset = build_dataset_manifest(
        structural,
        attestation,
        _artifact_digest("training-data"),
    )
    artifacts = {role: _artifact_digest(role) for role in REQUIRED_ROOT_ARTIFACTS}

    root_manifest = build_root_manifest(
        execution_id="execution-001",
        created_at_utc="2026-07-15T01:00:00Z",
        expires_at_utc="2026-07-16T01:00:00Z",
        repository_commit="f" * 40,
        artifacts=artifacts,
        holdout_groups=dataset["groups"]["holdout"],
    )

    assert root_manifest["schema_version"] == "a2v-bundle-manifest-v1"
    assert root_manifest["canonical_json_version"] == 1
    assert root_manifest["builder_version"] == "a2v-bundle-builder-v1"
    assert root_manifest["validator_version"] == "a2v-validator-v1"
    assert set(root_manifest["artifacts"]) == REQUIRED_ROOT_ARTIFACTS
    assert root_manifest["artifacts"]["plan"] == {
        "bytes": artifacts["plan"].bytes,
        "sha256": artifacts["plan"].sha256,
    }
    assert len(root_manifest["holdout_groups"]) == 5
    assert "bundle_id" not in root_manifest
    assert compute_bundle_id(root_manifest) == compute_bundle_id(
        dict(reversed(list(root_manifest.items())))
    )


@pytest.mark.parametrize("runtime_role", ["approval", "preflight", "ledger", "logs", "provider_state", "outputs"])
def test_root_manifest_rejects_runtime_only_artifact_roles(runtime_role: str) -> None:
    artifacts = {role: _artifact_digest(role) for role in REQUIRED_ROOT_ARTIFACTS}
    artifacts[runtime_role] = _artifact_digest(runtime_role)

    with pytest.raises(ValueError, match="artifact roles"):
        build_root_manifest(
            execution_id="execution-001",
            created_at_utc="2026-07-15T01:00:00Z",
            expires_at_utc="2026-07-16T01:00:00Z",
            repository_commit="f" * 40,
            artifacts=artifacts,
            holdout_groups=[],
        )


def test_root_manifest_requires_five_bound_holdout_groups() -> None:
    structural, attestation = _reports()
    dataset = build_dataset_manifest(
        structural,
        attestation,
        _artifact_digest("training-data"),
    )
    artifacts = {role: _artifact_digest(role) for role in REQUIRED_ROOT_ARTIFACTS}

    with pytest.raises(ValueError, match="at least five holdout groups"):
        build_root_manifest(
            execution_id="execution-001",
            created_at_utc="2026-07-15T01:00:00Z",
            expires_at_utc="2026-07-16T01:00:00Z",
            repository_commit="f" * 40,
            artifacts=artifacts,
            holdout_groups=dataset["groups"]["holdout"][:4],
        )


def test_bundle_id_excludes_self_hash() -> None:
    with pytest.raises(ValueError, match="must not be serialized"):
        compute_bundle_id({"bundle_id": "0" * 64})


def test_build_command_writes_only_content_addressed_bundle_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "private-run"
    candidates = run_dir / "candidates"
    control = run_dir / "control"
    validation = run_dir / "validation"
    candidates.mkdir(parents=True)
    control.mkdir()
    validation.mkdir()
    structural, attestation = _reports(candidate_root=candidates)
    (run_dir / "plan.md").write_text("private approved plan", encoding="utf-8")
    inputs = {
        control / "structural-report.json": structural,
        control / "quality-attestation.json": attestation,
        control / "standing-authorization.json": {"policy_id": "policy-001"},
        control / "price-evidence.json": {"rate_usd_per_step": "0.006"},
        control / "execution-config.json": {
            "execution_id": "execution-001",
            "created_at_utc": "2026-07-15T01:00:00Z",
            "expires_at_utc": "2026-07-16T01:00:00Z",
        },
        validation / "provider-validation-selection.json": {
            "group_ids": ["sample_011", "sample_012"]
        },
    }
    for path, value in inputs.items():
        path.write_bytes(canonical_json_bytes(value))

    completed = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), str(run_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert set(output) == {"bundle_id", "status"}
    assert output["status"] == "built"
    assert completed.stderr == ""
    bundle = run_dir / "bundle"
    assert sorted(path.name for path in bundle.iterdir()) == [
        "bundle-manifest.json",
        "dataset-manifest.json",
        "training-data.zip",
    ]
    root_manifest = json.loads((bundle / "bundle-manifest.json").read_text("utf-8"))
    assert output["bundle_id"] == compute_bundle_id(root_manifest)
    private_path_bytes = str(tmp_path).encode("utf-8")
    assert private_path_bytes not in (bundle / "dataset-manifest.json").read_bytes()
    assert private_path_bytes not in (bundle / "bundle-manifest.json").read_bytes()
    with zipfile.ZipFile(bundle / "training-data.zip") as archive:
        assert len(archive.namelist()) == 40
        assert all(not name.startswith("sample_01") or name.startswith("sample_010") for name in archive.namelist())


def test_build_command_sanitizes_parse_and_runtime_failures(tmp_path: Path) -> None:
    sensitive = "PRIVATE-MARKER-CREDENTIAL"
    parse_failure = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), str(tmp_path), sensitive],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_failure = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), str(tmp_path / sensitive)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert parse_failure.returncode == 2
    assert parse_failure.stdout == ""
    assert parse_failure.stderr == "A2V_BUNDLE_ARGUMENT_ERROR\n"
    assert runtime_failure.returncode == 2
    assert runtime_failure.stdout == ""
    assert runtime_failure.stderr == "A2V_BUNDLE_BUILD_FAILED\n"
    assert sensitive not in parse_failure.stderr + runtime_failure.stderr
