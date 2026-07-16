from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from .a2v_quality import validate_quality_and_splits
from .artifacts import (
    FileDigest,
    canonical_json_bytes,
    safe_relative_name,
    sha256_file,
)


DATASET_MANIFEST_SCHEMA = "a2v-dataset-manifest-v1"
ROOT_MANIFEST_SCHEMA = "a2v-bundle-manifest-v1"
ARCHIVE_BUILDER_VERSION = "a2v-bundle-builder-v1"
VALIDATOR_VERSION = "a2v-validator-v1"
FIXED_ARCHIVE_DATETIME = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR = (stat.S_IFREG | 0o600) << 16
MAX_ARCHIVE_MEMBERS = 400
MAX_ARCHIVE_UNCOMPRESSED_BYTES = (1 << 31) - 1
MAX_ARCHIVE_COMPRESSION_RATIO = 2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
# Bundle-visible identities are typed UUIDv4 values rendered as 32 lowercase
# hex digits. The type prefix is public; the token contains no human label.
UUID4_HEX = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
GROUP_BUNDLE_ID_PATTERN = re.compile(rf"grp_{UUID4_HEX}", re.ASCII)
DATASET_BUNDLE_ID_PATTERN = re.compile(rf"dset_{UUID4_HEX}", re.ASCII)
EXECUTION_BUNDLE_ID_PATTERN = re.compile(rf"exec_{UUID4_HEX}", re.ASCII)
REQUIRED_ROOT_ARTIFACT_ROLES = frozenset(
    {
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
)
ROOT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "canonical_json_version",
        "execution_id",
        "created_at_utc",
        "expires_at_utc",
        "builder_version",
        "validator_version",
        "repository_commit",
        "artifacts",
        "holdout_groups",
    }
)
LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")
CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3L5H2L")
END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
CANONICAL_ZIP_VERSION = 20
CANONICAL_ZIP_MADE_BY = (3 << 8) | CANONICAL_ZIP_VERSION
CANONICAL_DOS_TIME = 0
CANONICAL_DOS_DATE = 33


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _validate_member_name(value: Any) -> str:
    if type(value) is not str or not value or value in {".", ".."}:
        raise ValueError("archive member name must be a normalized relative root filename")
    try:
        safe_relative_name(value)
    except ValueError as exc:
        raise ValueError(
            "archive member name must be a normalized relative root filename"
        ) from exc
    if "/" in value:
        raise ValueError("archive member name must be a normalized relative root filename")
    return value


def _validate_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must contain a lowercase SHA-256")
    return value


def _validate_byte_count(value: Any, *, label: str, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} byte count must be a {qualifier} integer")
    return value


def _digest_payload(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, FileDigest):
        byte_count = value.bytes
        digest = value.sha256
    elif type(value) is dict:
        keys = set(value)
        if keys not in ({"bytes", "sha256"}, {"name", "bytes", "sha256"}):
            raise ValueError(f"{label} digest must contain only bytes and sha256")
        byte_count = value["bytes"]
        digest = value["sha256"]
    else:
        raise ValueError(f"{label} must be a file digest")
    return {
        "bytes": _validate_byte_count(byte_count, label=label),
        "sha256": _validate_sha256(digest, label=label),
    }


def _manifest_file_record(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"name", "bytes", "sha256"}:
        raise ValueError(f"{label} must be an exact file digest")
    return {
        "name": _validate_member_name(value["name"]),
        "bytes": _validate_byte_count(value["bytes"], label=label, allow_zero=False),
        "sha256": _validate_sha256(value["sha256"], label=label),
    }


def _expected_group_names(group_id: str) -> set[str]:
    return {
        f"{group_id}.txt",
        f"{group_id}_audio.wav",
        f"{group_id}_end.mp4",
        f"{group_id}_start.png",
    }


def _canonical_manifest_group(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"group_id", "files"}:
        raise ValueError(f"{label} must contain exactly group_id and files")
    group_id = value["group_id"]
    if (
        type(group_id) is not str
        or GROUP_BUNDLE_ID_PATTERN.fullmatch(group_id) is None
    ):
        raise ValueError(f"{label} must use a machine-generated opaque group ID")
    files_value = value["files"]
    if type(files_value) is not list or len(files_value) != 4:
        raise ValueError(f"{label} must contain exactly four file digests")
    files = [
        _manifest_file_record(item, label=f"{label} file {index}")
        for index, item in enumerate(files_value)
    ]
    names = [item["name"] for item in files]
    if len(set(names)) != len(names) or set(names) != _expected_group_names(group_id):
        raise ValueError(f"{label} does not contain the exact A2V group filenames")
    return {"group_id": group_id, "files": sorted(files, key=lambda item: item["name"])}


def _normalize_expected_members(
    expected_members: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(expected_members, Mapping):
        values: list[dict[str, Any]] = []
        for name, value in expected_members.items():
            if not isinstance(value, Mapping):
                raise ValueError("expected member digests must be objects")
            record = dict(value)
            record.setdefault("name", name)
            values.append(record)
    else:
        values = [dict(value) for value in expected_members]

    result: dict[str, dict[str, Any]] = {}
    casefolded: set[str] = set()
    for index, value in enumerate(values):
        record = _manifest_file_record(value, label=f"expected member {index}")
        name = record["name"]
        if name in result:
            raise ValueError("expected member set contains a duplicate name")
        folded = name.casefold()
        if folded in casefolded:
            raise ValueError("expected member set contains case-colliding names")
        casefolded.add(folded)
        result[name] = record
    return result


def _source_groups(groups: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for index, value in enumerate(groups):
        if type(value) is not dict or set(value) != {"group_id", "split", "files"}:
            raise ValueError(f"archive group {index} must contain group_id, split, and files")
        group_id = value["group_id"]
        if (
            type(group_id) is not str
            or GROUP_BUNDLE_ID_PATTERN.fullmatch(group_id) is None
        ):
            raise ValueError(
                f"archive group {index} must use a machine-generated opaque group ID"
            )
        if group_id in seen_group_ids:
            raise ValueError(f"archive groups contain duplicate group ID {group_id}")
        seen_group_ids.add(group_id)
        split = value["split"]
        if type(split) is not str or split not in {"train", "holdout"}:
            raise ValueError(f"archive group {group_id} split must be train or holdout")
        files_value = value["files"]
        if type(files_value) is not list or len(files_value) != 4:
            raise ValueError(f"archive group {group_id} must contain exactly four files")
        files = []
        for file_index, file_value in enumerate(files_value):
            if type(file_value) is not dict or set(file_value) != {
                "name",
                "path",
                "bytes",
                "sha256",
            }:
                raise ValueError(
                    f"archive group {group_id} file {file_index} must contain an exact source digest"
                )
            name = _validate_member_name(file_value["name"])
            path_value = file_value["path"]
            if not isinstance(path_value, (str, os.PathLike)):
                raise ValueError(f"archive member {name} source path is invalid")
            files.append(
                {
                    "name": name,
                    "path": Path(path_value),
                    "bytes": _validate_byte_count(
                        file_value["bytes"],
                        label=f"archive member {name}",
                        allow_zero=False,
                    ),
                    "sha256": _validate_sha256(
                        file_value["sha256"],
                        label=f"archive member {name}",
                    ),
                }
            )
        names = [item["name"] for item in files]
        if len(set(names)) != 4 or set(names) != _expected_group_names(group_id):
            raise ValueError(f"archive group {group_id} does not contain the exact A2V filenames")
        normalized.append(
            {"group_id": group_id, "split": split, "files": files}
        )
    return normalized


def _regular_source(path: Path, *, member_name: str) -> None:
    if _is_symlink_or_junction(path):
        raise ValueError(f"archive member {member_name} source must be a regular file")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError(f"archive member {member_name} source is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"archive member {member_name} source must be a regular file")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_exact(source: Any, byte_count: int) -> bytes:
    content = source.read(byte_count)
    if len(content) != byte_count:
        raise ValueError("training archive has non-canonical ZIP headers")
    return content


def _inspect_canonical_zip_layout(
    archive_path: Path,
    infos: list[zipfile.ZipInfo],
) -> None:
    with archive_path.open("rb") as source:
        _inspect_canonical_zip_layout_stream(source, infos)


def _inspect_canonical_zip_layout_stream(
    source: BinaryIO,
    infos: list[zipfile.ZipInfo],
) -> None:
    source.seek(0, os.SEEK_END)
    archive_size = source.tell()
    if archive_size < END_OF_CENTRAL_DIRECTORY.size:
        raise ValueError("training archive has non-canonical ZIP headers")
    eocd_offset = archive_size - END_OF_CENTRAL_DIRECTORY.size
    source.seek(eocd_offset)
    eocd = END_OF_CENTRAL_DIRECTORY.unpack(
        _read_exact(source, END_OF_CENTRAL_DIRECTORY.size)
    )
    if eocd[0] != END_OF_CENTRAL_DIRECTORY_SIGNATURE:
        raise ValueError("training archive contains a hidden trailing payload")
    (
        _,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = eocd
    if (
        disk_number != 0
        or central_disk != 0
        or disk_entries != len(infos)
        or total_entries != len(infos)
        or comment_length != 0
        or central_offset + central_size != eocd_offset
    ):
        raise ValueError("training archive has non-canonical ZIP headers")

    cursor = 0
    for info in infos:
        if info.header_offset != cursor:
            raise ValueError("training archive has non-canonical ZIP headers")
        source.seek(cursor)
        local = LOCAL_FILE_HEADER.unpack(
            _read_exact(source, LOCAL_FILE_HEADER.size)
        )
        (
            signature,
            extract_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
        ) = local
        raw_name = _read_exact(source, name_length)
        raw_extra = _read_exact(source, extra_length)
        expected_name = info.filename.encode("ascii")
        if (
            signature != LOCAL_FILE_SIGNATURE
            or extract_version != CANONICAL_ZIP_VERSION
            or flags != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != CANONICAL_DOS_TIME
            or modified_date != CANONICAL_DOS_DATE
            or crc != info.CRC
            or compressed_size != info.compress_size
            or uncompressed_size != info.file_size
            or raw_name != expected_name
            or raw_extra
        ):
            raise ValueError("training archive has non-canonical ZIP headers")
        cursor += LOCAL_FILE_HEADER.size + name_length + extra_length + compressed_size
    if cursor != central_offset:
        raise ValueError("training archive has non-canonical ZIP headers")

    cursor = central_offset
    for info in infos:
        source.seek(cursor)
        central = CENTRAL_DIRECTORY_HEADER.unpack(
            _read_exact(source, CENTRAL_DIRECTORY_HEADER.size)
        )
        (
            signature,
            made_by,
            extract_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            member_comment_length,
            start_disk,
            internal_attr,
            external_attr,
            local_offset,
        ) = central
        raw_name = _read_exact(source, name_length)
        raw_extra = _read_exact(source, extra_length)
        raw_comment = _read_exact(source, member_comment_length)
        expected_name = info.filename.encode("ascii")
        if (
            signature != CENTRAL_DIRECTORY_SIGNATURE
            or made_by != CANONICAL_ZIP_MADE_BY
            or extract_version != CANONICAL_ZIP_VERSION
            or flags != 0
            or compression != zipfile.ZIP_STORED
            or modified_time != CANONICAL_DOS_TIME
            or modified_date != CANONICAL_DOS_DATE
            or crc != info.CRC
            or compressed_size != info.compress_size
            or uncompressed_size != info.file_size
            or raw_name != expected_name
            or raw_extra
            or raw_comment
            or start_disk != 0
            or internal_attr != 0
            or external_attr != FIXED_EXTERNAL_ATTR
            or local_offset != info.header_offset
        ):
            raise ValueError("training archive has non-canonical ZIP headers")
        cursor += (
            CENTRAL_DIRECTORY_HEADER.size
            + name_length
            + extra_length
            + member_comment_length
        )
    if cursor != central_offset + central_size or cursor != eocd_offset:
        raise ValueError("training archive has non-canonical ZIP headers")


def _write_source_member(
    archive: zipfile.ZipFile,
    source: Mapping[str, Any],
) -> None:
    name = source["name"]
    path = source["path"]
    _regular_source(path, member_name=name)
    info = zipfile.ZipInfo(name, date_time=FIXED_ARCHIVE_DATETIME)
    info.create_system = 3
    info.external_attr = FIXED_EXTERNAL_ATTR
    info.internal_attr = 0
    info.compress_type = zipfile.ZIP_STORED
    info.flag_bits = 0
    info.extra = b""
    info.comment = b""
    info.file_size = source["bytes"]
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as input_file, archive.open(info, "w") as output_file:
        while chunk := input_file.read(1024 * 1024):
            output_file.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count != source["bytes"] or digest.hexdigest() != source["sha256"]:
        raise ValueError(f"archive member {name} does not match its structural digest")


def _declared_layout_requires_zip64(members: list[dict[str, Any]]) -> bool:
    if any(
        member["bytes"] * 105 > zipfile.ZIP64_LIMIT * 100
        for member in members
    ):
        return True
    central_offset = sum(
        LOCAL_FILE_HEADER.size + len(member["name"].encode("ascii")) + member["bytes"]
        for member in members
    )
    central_size = sum(
        CENTRAL_DIRECTORY_HEADER.size + len(member["name"].encode("ascii"))
        for member in members
    )
    return (
        central_offset > zipfile.ZIP64_LIMIT
        or central_size > zipfile.ZIP64_LIMIT
        or len(members) > zipfile.ZIP_FILECOUNT_LIMIT
    )


def build_training_archive(
    groups: Iterable[Mapping[str, Any]],
    destination: Path,
) -> FileDigest:
    """Build and reopen a deterministic train-only A2V ZIP archive."""

    destination = Path(destination)
    normalized_groups = _source_groups(groups)
    all_sources = [file for group in normalized_groups for file in group["files"]]
    members = [
        file
        for group in normalized_groups
        if group["split"] == "train"
        for file in group["files"]
    ]
    if not members:
        raise ValueError("training archive requires at least one training group")
    members.sort(key=lambda item: item["name"])
    names = [item["name"] for item in members]
    if len(set(names)) != len(names):
        raise ValueError("training archive contains duplicate member names")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("training archive contains case-colliding member names")
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("training archive exceeds the declared member-count limit")
    if sum(member["bytes"] for member in members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError("training archive exceeds the declared uncompressed-size limit")
    if _declared_layout_requires_zip64(members):
        raise ValueError("training archive has a prohibited declared ZIP64 requirement")
    if not destination.parent.is_dir() or _is_symlink_or_junction(destination.parent):
        raise ValueError("training archive destination parent must be a regular directory")
    if _is_symlink_or_junction(destination):
        raise ValueError("training archive destination must not be a link")
    for source in all_sources:
        if _same_path(destination, source["path"]):
            raise ValueError("training archive destination aliases a source member")

    expected = [
        {"name": item["name"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in members
    ]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
        ) as archive:
            archive.comment = b""
            for member in members:
                _write_source_member(archive, member)
        with temporary_path.open("r+b") as temporary:
            os.fsync(temporary.fileno())
        inspect_training_archive(temporary_path, expected)
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_parent(destination.parent)
        return inspect_training_archive(destination, expected)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def inspect_training_archive(
    archive_path: Path,
    expected_members: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    max_compression_ratio: int = MAX_ARCHIVE_COMPRESSION_RATIO,
) -> FileDigest:
    """Inspect one path-opened archive object against its manifest."""

    archive_path = Path(archive_path)
    try:
        with archive_path.open("rb") as source:
            with zipfile.ZipFile(source, mode="r") as archive:
                return inspect_open_training_archive(
                    archive,
                    expected_members,
                    name=archive_path.name,
                    max_members=max_members,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                    max_compression_ratio=max_compression_ratio,
                )
    except ValueError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("training archive is invalid") from exc


def _sha256_open_file(source: BinaryIO, *, name: str) -> FileDigest:
    position = source.tell()
    digest = hashlib.sha256()
    byte_count = 0
    try:
        source.seek(0)
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    finally:
        source.seek(position)
    return FileDigest(name=name, bytes=byte_count, sha256=digest.hexdigest())


def inspect_open_training_archive(
    archive: zipfile.ZipFile,
    expected_members: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    name: str,
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    max_compression_ratio: int = MAX_ARCHIVE_COMPRESSION_RATIO,
) -> FileDigest:
    """Inspect metadata and payloads through one retained ZIP/file object."""

    if type(max_members) is not int or max_members <= 0:
        raise ValueError("archive member-count limit must be a positive integer")
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
        raise ValueError("archive uncompressed-size limit must be a positive integer")
    if type(max_compression_ratio) is not int or max_compression_ratio <= 0:
        raise ValueError("archive compression-ratio limit must be a positive integer")
    expected = _normalize_expected_members(expected_members)
    source = archive.fp
    if source is None or not source.seekable() or not source.readable():
        raise ValueError("training archive requires one retained readable file object")

    try:
        if archive.comment:
            raise ValueError("training archive contains prohibited metadata")
        infos = archive.infolist()
        if len(infos) > max_members:
            raise ValueError("training archive exceeds the member-count limit")

        names: list[str] = []
        seen: set[str] = set()
        casefolded: set[str] = set()
        total_size = 0
        for info in infos:
            if (
                type(info.orig_filename) is not str
                or "\x00" in info.orig_filename
                or info.orig_filename != info.filename
            ):
                raise ValueError("training archive contains an invalid raw member name")
            _validate_member_name(info.orig_filename)
            member_name = _validate_member_name(info.filename)
            if member_name in seen:
                raise ValueError("training archive contains duplicate member names")
            folded = member_name.casefold()
            if folded in casefolded:
                raise ValueError("training archive contains case-colliding member names")
            seen.add(member_name)
            casefolded.add(folded)
            names.append(member_name)

            if info.flag_bits & 0x1:
                raise ValueError("training archive member encryption is prohibited")
            if info.flag_bits != 0:
                raise ValueError("training archive has non-canonical ZIP headers")
            if info.is_dir():
                raise ValueError("training archive members require regular-file attributes")
            unix_mode = info.external_attr >> 16
            if (
                info.create_system != 3
                or not stat.S_ISREG(unix_mode)
                or info.external_attr != FIXED_EXTERNAL_ATTR
            ):
                raise ValueError("training archive members require regular-file attributes")
            if info.file_size < 0 or info.compress_size < 0:
                raise ValueError("training archive contains invalid member sizes")
            total_size += info.file_size
            if total_size > max_uncompressed_bytes:
                raise ValueError("training archive exceeds the uncompressed-size limit")
            if info.file_size > max_compression_ratio * max(1, info.compress_size):
                raise ValueError("training archive exceeds the compression-ratio limit")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("training archive members must use ZIP_STORED")
            if info.compress_size != info.file_size:
                raise ValueError("training archive ZIP_STORED member has unequal stored sizes")
            if (
                info.date_time != FIXED_ARCHIVE_DATETIME
                or info.internal_attr != 0
                or info.extra
                or info.comment
            ):
                raise ValueError("training archive contains prohibited metadata")

        if names != sorted(names):
            raise ValueError("training archive member order is not lexical")
        _inspect_canonical_zip_layout_stream(source, infos)
        unexpected = sorted(set(names) - set(expected))
        if unexpected:
            raise ValueError("training archive contains unexpected members")
        missing = sorted(set(expected) - set(names))
        if missing:
            raise ValueError("training archive is missing members")

        for info in infos:
            expected_record = expected[info.filename]
            digest = hashlib.sha256()
            byte_count = 0
            with archive.open(info, mode="r") as member:
                while chunk := member.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
            if (
                byte_count != expected_record["bytes"]
                or digest.hexdigest() != expected_record["sha256"]
            ):
                raise ValueError(
                    f"archive member {info.filename} does not match its manifest"
                )
    except ValueError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("training archive is invalid") from exc
    return _sha256_open_file(source, name=name)


def _canonical_object_digest(value: Any) -> dict[str, Any]:
    content = canonical_json_bytes(value)
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _verify_candidate_file(
    candidate_dir: Path,
    record: Mapping[str, Any],
) -> None:
    path = candidate_dir / record["name"]
    _regular_source(path, member_name=record["name"])
    digest = sha256_file(path)
    if digest.bytes != record["bytes"] or digest.sha256 != record["sha256"]:
        raise ValueError(
            f"candidate member {record['name']} does not match its structural digest"
        )


def build_dataset_manifest(
    structural_report: dict[str, Any],
    quality_attestation: dict[str, Any],
    archive_digest: FileDigest | Mapping[str, Any],
    *,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    """Create the canonical train/holdout manifest without exposing provenance."""

    summary = validate_quality_and_splits(quality_attestation, structural_report)
    dataset_id = quality_attestation["dataset_id"]
    if (
        type(dataset_id) is not str
        or DATASET_BUNDLE_ID_PATTERN.fullmatch(dataset_id) is None
    ):
        raise ValueError("dataset_id must be a machine-generated opaque ID")
    structural_groups = {
        group["group_id"]: _canonical_manifest_group(
            group,
            label=f"structural group {group['group_id']}",
        )
        for group in structural_report["groups"]
    }
    train_groups = [
        structural_groups[group_id]
        for group_id in summary["accepted_train_group_ids"]
    ]
    holdout_groups = [
        structural_groups[group_id]
        for group_id in summary["accepted_holdout_group_ids"]
    ]
    if candidate_dir is not None:
        candidate_dir = Path(candidate_dir)
        if _is_symlink_or_junction(candidate_dir) or not candidate_dir.is_dir():
            raise ValueError("candidate root must be a regular directory")
        for group in train_groups + holdout_groups:
            for record in group["files"]:
                _verify_candidate_file(candidate_dir, record)

    training_members = sorted(
        (dict(record) for group in train_groups for record in group["files"]),
        key=lambda item: item["name"],
    )
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "spec": dict(structural_report["spec"]),
        "counts": {
            "train_groups": len(train_groups),
            "holdout_groups": len(holdout_groups),
        },
        "training_members": training_members,
        "groups": {
            "train": train_groups,
            "holdout": holdout_groups,
        },
        "reports": {
            "structural_report": _canonical_object_digest(structural_report),
            "quality_attestation": _canonical_object_digest(quality_attestation),
        },
        "archive": _digest_payload(archive_digest, label="training archive"),
        "archive_builder_version": ARCHIVE_BUILDER_VERSION,
    }
    canonical_json_bytes(manifest)
    return manifest


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be UTC")
    return parsed


def _canonical_holdout_groups(values: Any) -> list[dict[str, Any]]:
    if type(values) is not list:
        raise ValueError("holdout_groups must be a list")
    result = [
        _canonical_manifest_group(value, label=f"holdout group {index}")
        for index, value in enumerate(values)
    ]
    if len(result) < 5:
        raise ValueError("root manifest requires at least five holdout groups")
    group_ids = [group["group_id"] for group in result]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("holdout_groups contains duplicate group IDs")
    return sorted(result, key=lambda group: group["group_id"])


def _validate_serialized_root_manifest(root_manifest: Any) -> None:
    if type(root_manifest) is not dict or set(root_manifest) != ROOT_MANIFEST_KEYS:
        raise ValueError("root manifest must contain the exact top-level keys")
    if root_manifest["schema_version"] != ROOT_MANIFEST_SCHEMA:
        raise ValueError("root manifest has an unsupported schema_version")
    if (
        type(root_manifest["canonical_json_version"]) is not int
        or root_manifest["canonical_json_version"] != 1
    ):
        raise ValueError("root manifest has an unsupported canonical_json_version")
    if root_manifest["builder_version"] != ARCHIVE_BUILDER_VERSION:
        raise ValueError("root manifest has an unsupported builder_version")
    if root_manifest["validator_version"] != VALIDATOR_VERSION:
        raise ValueError("root manifest has an unsupported validator_version")

    execution_id = root_manifest["execution_id"]
    if (
        type(execution_id) is not str
        or EXECUTION_BUNDLE_ID_PATTERN.fullmatch(execution_id) is None
    ):
        raise ValueError("execution_id must be a machine-generated opaque ID")
    created = _utc_timestamp(root_manifest["created_at_utc"], label="created_at_utc")
    expires = _utc_timestamp(root_manifest["expires_at_utc"], label="expires_at_utc")
    if expires <= created:
        raise ValueError("expires_at_utc must be after created_at_utc")
    repository_commit = root_manifest["repository_commit"]
    if type(repository_commit) is not str or re.fullmatch(
        r"[0-9a-f]{40}", repository_commit, re.ASCII
    ) is None:
        raise ValueError("repository_commit must be a full lowercase Git commit")

    artifacts = root_manifest["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != REQUIRED_ROOT_ARTIFACT_ROLES:
        raise ValueError("root artifact roles must match the exact digest domain")
    for role in sorted(REQUIRED_ROOT_ARTIFACT_ROLES):
        digest = artifacts[role]
        if type(digest) is not dict or set(digest) != {"bytes", "sha256"}:
            raise ValueError(f"root artifact {role} must be an exact digest")
        _digest_payload(digest, label=f"root artifact {role}")

    holdout_groups = root_manifest["holdout_groups"]
    canonical_holdouts = _canonical_holdout_groups(holdout_groups)
    if holdout_groups != canonical_holdouts:
        raise ValueError("root manifest requires canonical holdout order")


def build_root_manifest(
    *,
    execution_id: str,
    created_at_utc: str,
    expires_at_utc: str,
    repository_commit: str,
    artifacts: Mapping[str, FileDigest | Mapping[str, Any]],
    holdout_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the exact root digest domain; runtime artifacts are not accepted."""

    if (
        type(execution_id) is not str
        or EXECUTION_BUNDLE_ID_PATTERN.fullmatch(execution_id) is None
    ):
        raise ValueError("execution_id must be a machine-generated opaque ID")
    created = _utc_timestamp(created_at_utc, label="created_at_utc")
    expires = _utc_timestamp(expires_at_utc, label="expires_at_utc")
    if expires <= created:
        raise ValueError("expires_at_utc must be after created_at_utc")
    if type(repository_commit) is not str or re.fullmatch(
        r"[0-9a-f]{40}", repository_commit, re.ASCII
    ) is None:
        raise ValueError("repository_commit must be a full lowercase Git commit")
    if not isinstance(artifacts, Mapping):
        raise ValueError("root artifacts must be a mapping")
    roles = set(artifacts)
    if roles != REQUIRED_ROOT_ARTIFACT_ROLES:
        raise ValueError("root artifact roles must match the exact digest domain")
    artifact_digests = {
        role: _digest_payload(artifacts[role], label=f"root artifact {role}")
        for role in sorted(REQUIRED_ROOT_ARTIFACT_ROLES)
    }
    manifest = {
        "schema_version": ROOT_MANIFEST_SCHEMA,
        "canonical_json_version": 1,
        "execution_id": execution_id,
        "created_at_utc": created_at_utc,
        "expires_at_utc": expires_at_utc,
        "builder_version": ARCHIVE_BUILDER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "repository_commit": repository_commit,
        "artifacts": artifact_digests,
        "holdout_groups": _canonical_holdout_groups(holdout_groups),
    }
    _validate_serialized_root_manifest(manifest)
    canonical_json_bytes(manifest)
    return manifest


def compute_bundle_id(root_manifest: dict[str, Any]) -> str:
    if type(root_manifest) is dict and "bundle_id" in root_manifest:
        raise ValueError("bundle_id must not be serialized into its digest domain")
    _validate_serialized_root_manifest(root_manifest)
    return hashlib.sha256(canonical_json_bytes(root_manifest)).hexdigest()
