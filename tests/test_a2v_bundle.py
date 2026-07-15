from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest

import ltx_lora_pilot.a2v_bundle as a2v_bundle
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
DATASET_ID = "dset_0123456789ab4def8123456789abcdef"
EXECUTION_ID = "exec_fedcba9876544321a0fedcba98765432"


def _machine_group_id(index: int) -> str:
    return f"grp_000000000000400080000000{index:08x}"


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
        _make_group(tmp_path, _machine_group_id(2), "train"),
        _make_group(tmp_path, _machine_group_id(3), "holdout"),
        _make_group(tmp_path, _machine_group_id(1), "train"),
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


def _patch_zip_names(path: Path, *, local_name: bytes, central_name: bytes) -> None:
    content = bytearray(path.read_bytes())
    local_offset = content.index(b"PK\x03\x04")
    central_offset = content.index(b"PK\x01\x02")
    local_length = struct.unpack_from("<H", content, local_offset + 26)[0]
    central_length = struct.unpack_from("<H", content, central_offset + 28)[0]
    assert len(local_name) == local_length
    assert len(central_name) == central_length
    content[local_offset + 30 : local_offset + 30 + local_length] = local_name
    content[central_offset + 46 : central_offset + 46 + central_length] = central_name
    path.write_bytes(content)


def _patch_zip_flags(
    path: Path,
    *,
    local_flags: int | None = None,
    central_flags: int | None = None,
) -> None:
    content = bytearray(path.read_bytes())
    if local_flags is not None:
        local_offset = content.index(b"PK\x03\x04")
        struct.pack_into("<H", content, local_offset + 6, local_flags)
    if central_flags is not None:
        central_offset = content.index(b"PK\x01\x02")
        struct.pack_into("<H", content, central_offset + 8, central_flags)
    path.write_bytes(content)


def _patch_zip_uncompressed_size(path: Path, size: int) -> None:
    content = bytearray(path.read_bytes())
    local_offset = content.index(b"PK\x03\x04")
    central_offset = content.index(b"PK\x01\x02")
    struct.pack_into("<L", content, local_offset + 22, size)
    struct.pack_into("<L", content, central_offset + 24, size)
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
        group_id = _machine_group_id(index + 1)
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
        "dataset_id": DATASET_ID,
        "rights_and_consent": {
            "confirmed": True,
            "reviewer_id": "reviewer-opaque-001",
            "reviewed_at_utc": "2026-07-15T00:00:00Z",
        },
        "groups": list(reversed(quality_groups)),
    }
    return structural, attestation


def _valid_root_manifest() -> dict[str, Any]:
    structural, attestation = _reports()
    dataset = build_dataset_manifest(
        structural,
        attestation,
        _artifact_digest("training-data"),
    )
    artifacts = {role: _artifact_digest(role) for role in REQUIRED_ROOT_ARTIFACTS}
    return build_root_manifest(
        execution_id=EXECUTION_ID,
        created_at_utc="2026-07-15T01:00:00Z",
        expires_at_utc="2026-07-16T01:00:00Z",
        repository_commit="f" * 40,
        artifacts=artifacts,
        holdout_groups=dataset["groups"]["holdout"],
    )


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
        assert all(not name.startswith(_machine_group_id(3)) for name in names)


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


def test_inspection_rejects_raw_nul_member_name(tmp_path: Path) -> None:
    path = tmp_path / "nul-name.zip"
    content = b"content"
    _write_zip(path, [("safe.txt", content)])
    _patch_zip_names(path, local_name=b"safe\x00txt", central_name=b"safe\x00txt")

    with pytest.raises(ValueError, match="raw member name"):
        inspect_training_archive(path, [_digest_record("safe", content)])


def test_inspection_rejects_hidden_trailing_payload(tmp_path: Path) -> None:
    path = tmp_path / "trailing.zip"
    expected = _write_zip(path, [("sample.txt", b"content")])
    with path.open("ab") as archive:
        archive.write(b"HIDDEN-TRAILING-PAYLOAD")

    with pytest.raises(ValueError, match="trailing payload"):
        inspect_training_archive(path, expected)


def test_inspection_requires_equal_stored_sizes(tmp_path: Path) -> None:
    path = tmp_path / "stored-size.zip"
    content = b"12345678"
    expected = _write_zip(path, [("sample.txt", content)])
    _patch_zip_uncompressed_size(path, len(content) - 1)

    with pytest.raises(ValueError, match="stored sizes"):
        inspect_training_archive(path, expected)


@pytest.mark.parametrize(
    ("local_flags", "central_flags"),
    [(0x800, 0), (0x800, 0x800)],
)
def test_inspection_requires_canonical_ascii_header_flags(
    tmp_path: Path,
    local_flags: int,
    central_flags: int,
) -> None:
    path = tmp_path / "flags.zip"
    expected = _write_zip(path, [("sample.txt", b"content")])
    _patch_zip_flags(
        path,
        local_flags=local_flags,
        central_flags=central_flags,
    )

    with pytest.raises(ValueError, match="canonical ZIP headers"):
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


def test_builder_rejects_human_readable_group_id(tmp_path: Path) -> None:
    groups = [_make_group(tmp_path, "human_readable_label", "train")]

    with pytest.raises(ValueError, match="machine-generated opaque group ID"):
        build_training_archive(groups, tmp_path / "training.zip")


@pytest.mark.parametrize("use_hardlink", [False, True])
def test_builder_rejects_destination_aliasing_holdout_source(
    tmp_path: Path,
    use_hardlink: bool,
) -> None:
    groups = _fixture_groups(tmp_path)
    holdout = next(group for group in groups if group["split"] == "holdout")
    source = holdout["files"][0]["path"]
    original = source.read_bytes()
    destination = source
    if use_hardlink:
        destination = tmp_path / "holdout-alias.zip"
        os.link(source, destination)

    with pytest.raises(ValueError, match="aliases a source member"):
        build_training_archive(groups, destination)

    assert source.read_bytes() == original
    assert destination.read_bytes() == original


def test_builder_prechecks_member_limit_before_replacing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "training.zip"
    previous = b"previous-reviewed-archive"
    destination.write_bytes(previous)
    groups = [
        _make_group(tmp_path, _machine_group_id(index), "train")
        for index in range(1, 102)
    ]

    with pytest.raises(ValueError, match="declared member-count limit"):
        build_training_archive(groups, destination)

    assert destination.read_bytes() == previous


def test_builder_prechecks_declared_size_before_reading_or_replacing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "training.zip"
    previous = b"previous-reviewed-archive"
    destination.write_bytes(previous)
    groups = _fixture_groups(tmp_path)
    train = next(group for group in groups if group["split"] == "train")
    train["files"][0]["bytes"] = a2v_bundle.MAX_ARCHIVE_UNCOMPRESSED_BYTES + 1

    with pytest.raises(ValueError, match="declared uncompressed-size limit"):
        build_training_archive(groups, destination)

    assert destination.read_bytes() == previous


@pytest.mark.parametrize("limit_case", ["member", "aggregate"])
def test_builder_prechecks_declared_zip64_requirement(
    tmp_path: Path,
    limit_case: str,
) -> None:
    destination = tmp_path / "training.zip"
    previous = b"previous-reviewed-archive"
    destination.write_bytes(previous)
    groups = [_make_group(tmp_path, _machine_group_id(1), "train")]
    files = groups[0]["files"]
    if limit_case == "member":
        files[0]["bytes"] = (zipfile.ZIP64_LIMIT * 100 // 105) + 1
    else:
        unchanged_bytes = sum(record["bytes"] for record in files[2:])
        files[0]["bytes"] = a2v_bundle.MAX_ARCHIVE_UNCOMPRESSED_BYTES // 2
        files[1]["bytes"] = (
            a2v_bundle.MAX_ARCHIVE_UNCOMPRESSED_BYTES
            - files[0]["bytes"]
            - unchanged_bytes
        )

    with pytest.raises(ValueError, match="declared ZIP64 requirement"):
        build_training_archive(groups, destination)

    assert destination.read_bytes() == previous


def test_builder_inspects_temporary_archive_before_final_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "training.zip"
    destination.write_bytes(b"previous-reviewed-archive")
    groups = _fixture_groups(tmp_path)
    real_inspector = a2v_bundle.inspect_training_archive
    inspected: list[Path] = []

    def recording_inspector(
        archive_path: Path,
        expected_members: Any,
        **limits: Any,
    ) -> FileDigest:
        result = real_inspector(archive_path, expected_members, **limits)
        inspected.append(Path(archive_path))
        return result

    monkeypatch.setattr(a2v_bundle, "inspect_training_archive", recording_inspector)

    build_training_archive(groups, destination)

    assert len(inspected) == 2
    assert inspected[0].parent == destination.parent
    assert inspected[0] != destination
    assert inspected[1] == destination


def test_dataset_manifest_binds_train_holdout_reports_and_archive() -> None:
    structural, attestation = _reports()
    archive = _artifact_digest("training-data")

    manifest = build_dataset_manifest(structural, attestation, archive)

    structural_bytes = canonical_json_bytes(structural)
    attestation_bytes = canonical_json_bytes(attestation)
    assert manifest["schema_version"] == "a2v-dataset-manifest-v1"
    assert manifest["dataset_id"] == DATASET_ID
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
    holdout_name = f"{_machine_group_id(15)}_end.mp4"
    (tmp_path / holdout_name).write_bytes(b"changed-holdout")

    with pytest.raises(ValueError, match="does not match its structural digest"):
        build_dataset_manifest(
            structural,
            attestation,
            _artifact_digest("training-data"),
            candidate_dir=tmp_path,
        )


def test_dataset_manifest_rejects_human_readable_dataset_id() -> None:
    structural, attestation = _reports()
    attestation["dataset_id"] = "human_readable_label"

    with pytest.raises(ValueError, match="machine-generated opaque ID"):
        build_dataset_manifest(
            structural,
            attestation,
            _artifact_digest("training-data"),
        )


def test_dataset_manifest_rejects_human_readable_group_id() -> None:
    structural, attestation = _reports()
    structural_group = structural["groups"][0]
    old_group_id = structural_group["group_id"]
    structural_group["group_id"] = "human_readable_label"
    for record in structural_group["files"]:
        record["name"] = record["name"].replace(
            old_group_id,
            "human_readable_label",
            1,
        )
    attestation_group = next(
        group for group in attestation["groups"] if group["group_id"] == old_group_id
    )
    attestation_group["group_id"] = "human_readable_label"

    with pytest.raises(ValueError, match="machine-generated opaque group ID"):
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
        execution_id=EXECUTION_ID,
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
            execution_id=EXECUTION_ID,
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
            execution_id=EXECUTION_ID,
            created_at_utc="2026-07-15T01:00:00Z",
            expires_at_utc="2026-07-16T01:00:00Z",
            repository_commit="f" * 40,
            artifacts=artifacts,
            holdout_groups=dataset["groups"]["holdout"][:4],
        )


def test_root_manifest_rejects_human_readable_execution_id() -> None:
    structural, attestation = _reports()
    dataset = build_dataset_manifest(
        structural,
        attestation,
        _artifact_digest("training-data"),
    )
    artifacts = {role: _artifact_digest(role) for role in REQUIRED_ROOT_ARTIFACTS}

    with pytest.raises(ValueError, match="machine-generated opaque ID"):
        build_root_manifest(
            execution_id="human_readable_label",
            created_at_utc="2026-07-15T01:00:00Z",
            expires_at_utc="2026-07-16T01:00:00Z",
            repository_commit="f" * 40,
            artifacts=artifacts,
            holdout_groups=dataset["groups"]["holdout"],
        )


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_compute_bundle_id_rejects_non_exact_top_level_schema(mutation: str) -> None:
    root_manifest = _valid_root_manifest()
    if mutation == "unknown":
        root_manifest["approval"] = {"status": "runtime-only"}
    else:
        del root_manifest["created_at_utc"]

    with pytest.raises(ValueError, match="exact top-level keys"):
        compute_bundle_id(root_manifest)


def test_compute_bundle_id_rejects_unknown_nested_artifact_role() -> None:
    root_manifest = _valid_root_manifest()
    root_manifest["artifacts"]["logs"] = {
        "bytes": 0,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="artifact roles"):
        compute_bundle_id(root_manifest)


def test_compute_bundle_id_rejects_unknown_nested_digest_field() -> None:
    root_manifest = _valid_root_manifest()
    root_manifest["artifacts"]["plan"]["provider_state"] = "runtime-only"

    with pytest.raises(ValueError, match="exact digest"):
        compute_bundle_id(root_manifest)


def test_compute_bundle_id_rejects_unknown_nested_holdout_field() -> None:
    root_manifest = _valid_root_manifest()
    root_manifest["holdout_groups"][0]["outputs"] = []

    with pytest.raises(ValueError, match="holdout group"):
        compute_bundle_id(root_manifest)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", "a2v-bundle-manifest-v2"),
        ("canonical_json_version", 2),
        ("builder_version", "a2v-bundle-builder-v2"),
        ("validator_version", "a2v-validator-v2"),
    ],
)
def test_compute_bundle_id_rejects_unsupported_root_versions(
    field: str,
    invalid_value: Any,
) -> None:
    root_manifest = _valid_root_manifest()
    root_manifest[field] = invalid_value

    with pytest.raises(ValueError, match="root manifest"):
        compute_bundle_id(root_manifest)


def test_compute_bundle_id_rejects_human_readable_holdout_group_id() -> None:
    root_manifest = _valid_root_manifest()
    root_manifest["holdout_groups"][0]["group_id"] = "human_readable_label"

    with pytest.raises(ValueError, match="machine-generated opaque group ID"):
        compute_bundle_id(root_manifest)


def test_compute_bundle_id_requires_canonical_holdout_order() -> None:
    root_manifest = _valid_root_manifest()
    root_manifest["holdout_groups"].reverse()

    with pytest.raises(ValueError, match="canonical holdout order"):
        compute_bundle_id(root_manifest)


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
            "execution_id": EXECUTION_ID,
            "created_at_utc": "2026-07-15T01:00:00Z",
            "expires_at_utc": "2026-07-16T01:00:00Z",
        },
        validation / "provider-validation-selection.json": {
            "group_ids": [_machine_group_id(11), _machine_group_id(12)]
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
        expected_names = {
            record["name"]
            for group in structural["groups"]
            if group["group_id"] in {_machine_group_id(index) for index in range(1, 11)}
            for record in group["files"]
        }
        assert set(archive.namelist()) == expected_names


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
