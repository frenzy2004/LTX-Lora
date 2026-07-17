"""Offline immutable-provenance verification for canonical A2V bundles.

This module deliberately excludes provider clients, credential resolution,
execution receipts, and ledger implementations. It owns the static integrity
prefix shared by the fresh-run issuer and dynamic paid-execution preflight.
"""

from __future__ import annotations

import copy
import ctypes
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, BinaryIO, Callable, Mapping
import zipfile

from .a2v_bundle import (
    ARCHIVE_BUILDER_VERSION,
    DATASET_MANIFEST_SCHEMA,
    ROOT_MANIFEST_KEYS,
    ROOT_MANIFEST_SCHEMA,
    REQUIRED_ROOT_ARTIFACT_ROLES,
    build_dataset_manifest,
    compute_bundle_id,
    inspect_open_training_archive,
)
from .a2v_contracts import (
    APPROVAL_SCHEMA_VERSION,
    EXECUTION_CONFIG_FIELDS,
    EXECUTION_CONFIG_SCHEMA_VERSION,
    EXECUTION_RECEIPT_FIELDS,
    PRICE_EVIDENCE_FIELDS,
    SHA256_PATTERN,
    STANDING_AUTHORIZATION_FIELDS,
    validate_execution_config,
)
from .a2v_dataset import A2VSpec, STRUCTURAL_REPORT_SCHEMA, validate_a2v_directory
from .a2v_quality import (
    ATTESTATION_KEYS,
    QUALITY_ATTESTATION_SCHEMA,
    validate_quality_and_splits,
)
from .artifacts import (
    FileDigest,
    canonical_json_bytes,
    safe_relative_name,
    strict_load_json,
)
from .private_workspace import (
    EXECUTION_ID_PATTERN,
    PILOT_ID_PATTERN,
    require_canonical_run_dir,
)
from .provider_validation import (
    CANONICAL_JSON_VERSION as PROVIDER_CANONICAL_JSON_VERSION,
    DIGEST_FIELDS as PROVIDER_DIGEST_FIELDS,
    ITEM_FIELDS as PROVIDER_ITEM_FIELDS,
    SCHEMA_VERSION as PROVIDER_SELECTION_SCHEMA,
    SELECTION_FIELDS as PROVIDER_SELECTION_FIELDS,
    validate_provider_validation_selection,
)


DATASET_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "spec",
        "counts",
        "training_members",
        "groups",
        "reports",
        "archive",
        "archive_builder_version",
    }
)
STRUCTURAL_REPORT_FIELDS = frozenset({"schema_version", "status", "spec", "groups"})
ROOT_ARTIFACT_PATHS = {
    "plan": Path("plan.md"),
    "standing_authorization": Path("control/standing-authorization.json"),
    "price_evidence": Path("control/price-evidence.json"),
    "structural_report": Path("control/structural-report.json"),
    "quality_attestation": Path("control/quality-attestation.json"),
    "dataset_manifest": Path("bundle/dataset-manifest.json"),
    "training_archive": Path("bundle/training-data.zip"),
    "execution_config": Path("control/execution-config.json"),
    "provider_validation_selection": Path(
        "validation/provider-validation-selection.json"
    ),
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _StaticGateFailure(RuntimeError):
    def __init__(self, gate: str, passed_gates: tuple[str, ...]) -> None:
        super().__init__(gate)
        self.gate = gate
        self.passed_gates = passed_gates


@dataclass(frozen=True)
class StaticA2VBundle:
    """Detached immutable-provenance snapshot of a canonical A2V bundle."""

    private_root: Path
    run_dir: Path
    pilot_id: str
    execution_id: str
    bundle_id: str
    root_manifest: dict[str, Any]
    structural_report: dict[str, Any]
    quality_attestation: dict[str, Any]
    dataset_manifest: dict[str, Any]
    provider_selection: dict[str, Any]
    execution_config: dict[str, Any]
    price_evidence: dict[str, Any]
    standing_policy: dict[str, Any]
    quality_summary: dict[str, Any]
    archive_digest: FileDigest
    passed_gates: tuple[str, ...]


@dataclass(frozen=True)
class _NodeFingerprint:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _PinnedFile:
    path: Path = field(repr=False)
    fingerprint: _NodeFingerprint
    bytes: int
    sha256: str


@dataclass
class _OpenPinnedArchive:
    source: BinaryIO = field(repr=False)
    archive: zipfile.ZipFile = field(repr=False)
    pin: _PinnedFile
    digest: FileDigest

    def close(self) -> None:
        try:
            self.archive.close()
        finally:
            self.source.close()


@dataclass(frozen=True)
class _LoadedArtifacts:
    root_manifest: dict[str, Any]
    structural_report: dict[str, Any]
    quality_attestation: dict[str, Any]
    dataset_manifest: dict[str, Any]
    provider_selection: dict[str, Any]
    execution_config: dict[str, Any]
    price_evidence: dict[str, Any]
    standing_policy: dict[str, Any]
    receipt: dict[str, Any] | None


@dataclass(frozen=True)
class _PathContext:
    private_root: Path
    run_dir: Path
    pilot_id: str
    execution_id: str
    security_snapshot: dict[str, _NodeFingerprint] = field(repr=False)


@dataclass(frozen=True)
class _StaticVerification:
    context: _PathContext
    artifacts: _LoadedArtifacts
    pins: dict[str, _PinnedFile] = field(repr=False)
    bundle_id: str
    archive_digest: FileDigest
    fresh_structural: dict[str, Any]
    quality_summary: dict[str, Any]
    rebuilt_manifest: dict[str, Any]
    provider_selection: dict[str, Any]
    config: dict[str, Any]
    provider_items: int
    passed_gates: tuple[str, ...]


def _node_fingerprint(value: os.stat_result) -> _NodeFingerprint:
    is_directory = stat.S_ISDIR(value.st_mode)
    return _NodeFingerprint(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        links=int(value.st_nlink),
        size=0 if is_directory else int(value.st_size),
        mtime_ns=0 if is_directory else int(value.st_mtime_ns),
    )


def _is_reparse_or_link(path: Path, metadata: os.stat_result | None = None) -> bool:
    try:
        current = path.lstat() if metadata is None else metadata
        attributes = getattr(current, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        is_junction = getattr(os.path, "isjunction", None)
        return (
            stat.S_ISLNK(current.st_mode)
            or bool(attributes & reparse_flag)
            or bool(is_junction is not None and is_junction(path))
        )
    except OSError:
        return True


def _has_ads_syntax(path: Path) -> bool:
    parts = path.parts[1:] if path.anchor else path.parts
    return any(":" in part for part in parts)


def _case_is_exact(path: Path) -> bool:
    if os.name != "nt":
        return True
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        try:
            with os.scandir(current) as entries:
                matches = [entry.name for entry in entries if entry.name.casefold() == part.casefold()]
        except OSError:
            return False
        if matches != [part]:
            return False
        current /= part
    return True


def _default_windows_dacl_check(path: Path) -> None:
    if os.name != "nt":
        return
    from ctypes import wintypes

    class TRUSTEE_W(ctypes.Structure):
        pass

    TRUSTEE_W._fields_ = [
        ("pMultipleTrustee", ctypes.POINTER(TRUSTEE_W)),
        ("MultipleTrusteeOperation", wintypes.DWORD),
        ("TrusteeForm", wintypes.DWORD),
        ("TrusteeType", wintypes.DWORD),
        ("ptstrName", ctypes.c_void_p),
    ]

    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_security.restype = wintypes.DWORD
    convert_sid = advapi32.ConvertStringSidToSidW
    convert_sid.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    convert_sid.restype = wintypes.BOOL
    effective_rights = advapi32.GetEffectiveRightsFromAclW
    effective_rights.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(TRUSTEE_W),
        ctypes.POINTER(wintypes.DWORD),
    ]
    effective_rights.restype = wintypes.DWORD
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = get_security(
        str(path),
        1,
        0x00000004,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if status != 0 or not dacl.value or not descriptor.value:
        if descriptor.value:
            local_free(descriptor)
        raise ValueError("private path DACL is unavailable")
    dangerous = 0x00120089 | 0x00120116 | 0x00010000 | 0x00040000 | 0x00080000
    try:
        for sid_text in ("S-1-1-0", "S-1-5-11", "S-1-5-32-545"):
            sid = ctypes.c_void_p()
            if not convert_sid(sid_text, ctypes.byref(sid)) or not sid.value:
                raise ValueError("private path DACL is unavailable")
            try:
                trustee = TRUSTEE_W(None, 0, 0, 5, sid)
                mask = wintypes.DWORD()
                result = effective_rights(dacl, ctypes.byref(trustee), ctypes.byref(mask))
                if result != 0:
                    raise ValueError("private path DACL is unavailable")
                if mask.value & dangerous:
                    raise ValueError("private path DACL grants broad access")
            finally:
                local_free(sid)
    finally:
        local_free(descriptor)


_WINDOWS_DACL_CHECK: Callable[[Path], None] = _default_windows_dacl_check


def _protected_metadata(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("private path is unavailable") from exc
    if _is_reparse_or_link(path, metadata):
        raise ValueError("private path aliases are prohibited")
    is_directory = stat.S_ISDIR(metadata.st_mode)
    is_regular = stat.S_ISREG(metadata.st_mode)
    if not is_directory and not is_regular:
        raise ValueError("private path must be a regular file or directory")
    if is_regular and metadata.st_nlink != 1:
        raise ValueError("private files must have one hard link")
    if os.name != "nt":
        permissions = stat.S_IMODE(metadata.st_mode)
        if is_directory and permissions != 0o700:
            raise ValueError("private directories must be owner-only")
        if is_regular and permissions & 0o077:
            raise ValueError("private files must be owner-only")
    _WINDOWS_DACL_CHECK(path)
    return metadata


def _ancestor_paths(path: Path) -> list[Path]:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    result = [current]
    for part in absolute.parts[1:]:
        current /= part
        result.append(current)
    return result


def _security_snapshot(private_root: Path) -> dict[str, _NodeFingerprint]:
    if not private_root.is_absolute() or _has_ads_syntax(private_root):
        raise ValueError("approved private root is invalid")
    for ancestor in _ancestor_paths(private_root):
        try:
            metadata = ancestor.lstat()
        except OSError as exc:
            raise ValueError("approved private root is unavailable") from exc
        if _is_reparse_or_link(ancestor, metadata) or not _case_is_exact(ancestor):
            raise ValueError("approved private root aliases are prohibited")
    snapshot: dict[str, _NodeFingerprint] = {}
    pending = [private_root]
    while pending:
        path = pending.pop()
        if _has_ads_syntax(path) or not _case_is_exact(path):
            raise ValueError("private path aliases are prohibited")
        metadata = _protected_metadata(path)
        snapshot[str(path)] = _node_fingerprint(metadata)
        if stat.S_ISDIR(metadata.st_mode):
            try:
                children = sorted(path.iterdir(), key=lambda item: item.name, reverse=True)
            except OSError as exc:
                raise ValueError("private directory is unavailable") from exc
            pending.extend(children)
    return snapshot


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve(strict=True).relative_to(Path(parent).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _lexical_context(
    private_root: Path,
    run_dir: Path,
) -> _PathContext:
    root = Path(private_root)
    run = Path(run_dir)
    for value in (root, run):
        raw = str(value)
        if (
            not raw
            or raw != raw.strip()
            or "\x00" in raw
            or not value.is_absolute()
            or ".." in value.parts
            or str(value) != os.path.abspath(value)
            or _has_ads_syntax(value)
        ):
            raise ValueError("canonical private path is required")
    root_parts = root.parts
    run_parts = run.parts
    if len(run_parts) != len(root_parts) + 4 or run_parts[: len(root_parts)] != root_parts:
        raise ValueError("canonical run directory is required")
    relative = run_parts[len(root_parts) :]
    if relative[0] != "pilots" or relative[2] != "runs":
        raise ValueError("canonical run directory is required")
    pilot_id, execution_id = relative[1], relative[3]
    if PILOT_ID_PATTERN.fullmatch(pilot_id) is None or EXECUTION_ID_PATTERN.fullmatch(execution_id) is None:
        raise ValueError("canonical run directory is required")
    if _is_within(root, REPOSITORY_ROOT) or _is_within(REPOSITORY_ROOT, root):
        raise ValueError("private root and repository must be separate")
    security = _security_snapshot(root)
    canonical_run = require_canonical_run_dir(root, pilot_id, execution_id, run)
    return _PathContext(
        private_root=root.resolve(strict=True),
        run_dir=canonical_run,
        pilot_id=pilot_id,
        execution_id=execution_id,
        security_snapshot=security,
    )


def _canonical_object(path: Path) -> dict[str, Any]:
    value = strict_load_json(path)
    if type(value) is not dict or path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("private JSON artifact is not canonical")
    return value


def _exact_fields(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} has unknown or missing fields")
    return value


def _load_canonical_artifacts(context: _PathContext, require_receipt: bool) -> _LoadedArtifacts:
    run = context.run_dir
    root = _canonical_object(run / "bundle" / "bundle-manifest.json")
    structural = _canonical_object(run / "control" / "structural-report.json")
    quality = _canonical_object(run / "control" / "quality-attestation.json")
    dataset = _canonical_object(run / "bundle" / "dataset-manifest.json")
    selection = _canonical_object(run / "validation" / "provider-validation-selection.json")
    config = _canonical_object(run / "control" / "execution-config.json")
    price = _canonical_object(run / "control" / "price-evidence.json")
    policy = _canonical_object(run / "control" / "standing-authorization.json")
    receipt_path = run / "control" / "execution-approval.json"
    receipt = _canonical_object(receipt_path) if require_receipt and receipt_path.is_file() else None

    _exact_fields(root, ROOT_MANIFEST_KEYS, label="root manifest")
    if root.get("schema_version") != ROOT_MANIFEST_SCHEMA:
        raise ValueError("root manifest schema is unsupported")
    _exact_fields(structural, STRUCTURAL_REPORT_FIELDS, label="structural report")
    if structural.get("schema_version") != STRUCTURAL_REPORT_SCHEMA:
        raise ValueError("structural report schema is unsupported")
    _exact_fields(quality, ATTESTATION_KEYS, label="quality attestation")
    if quality.get("schema_version") != QUALITY_ATTESTATION_SCHEMA:
        raise ValueError("quality attestation schema is unsupported")
    _exact_fields(dataset, DATASET_MANIFEST_FIELDS, label="dataset manifest")
    if dataset.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("dataset manifest schema is unsupported")
    if dataset.get("archive_builder_version") != ARCHIVE_BUILDER_VERSION:
        raise ValueError("dataset archive builder is unsupported")
    _exact_fields(selection, PROVIDER_SELECTION_FIELDS, label="provider selection")
    if (
        selection.get("schema_version") != PROVIDER_SELECTION_SCHEMA
        or selection.get("canonical_json_version") != PROVIDER_CANONICAL_JSON_VERSION
    ):
        raise ValueError("provider selection schema is unsupported")
    _exact_fields(config, EXECUTION_CONFIG_FIELDS, label="execution config")
    if config.get("schema_version") != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise ValueError("execution config schema is unsupported")
    _exact_fields(price, PRICE_EVIDENCE_FIELDS, label="price evidence")
    _exact_fields(policy, STANDING_AUTHORIZATION_FIELDS, label="standing policy")
    if receipt is not None:
        _exact_fields(receipt, EXECUTION_RECEIPT_FIELDS, label="execution receipt")
        if receipt.get("schema_version") != APPROVAL_SCHEMA_VERSION:
            raise ValueError("execution receipt schema is unsupported")
    return _LoadedArtifacts(
        root_manifest=root,
        structural_report=structural,
        quality_attestation=quality,
        dataset_manifest=dataset,
        provider_selection=selection,
        execution_config=config,
        price_evidence=price,
        standing_policy=policy,
        receipt=receipt,
    )


def _pin_file(path: Path) -> _PinnedFile:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("private artifact is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("private artifact must be a single-link regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        on_path = path.lstat()
    except OSError as exc:
        raise ValueError("private artifact changed during inspection") from exc
    if _is_reparse_or_link(path, on_path):
        raise ValueError("private artifact aliases are prohibited")
    identity = _node_fingerprint(before)
    if identity != _node_fingerprint(after) or identity != _node_fingerprint(on_path):
        raise ValueError("private artifact changed during inspection")
    return _PinnedFile(
        path=path,
        fingerprint=identity,
        bytes=byte_count,
        sha256=digest.hexdigest(),
    )


def _digest_open_file(source: BinaryIO, pin: _PinnedFile) -> _PinnedFile:
    before = os.fstat(source.fileno())
    digest = hashlib.sha256()
    byte_count = 0
    position = source.tell()
    try:
        source.seek(0)
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    finally:
        source.seek(position)
    after = os.fstat(source.fileno())
    before_fingerprint = _node_fingerprint(before)
    if before_fingerprint != _node_fingerprint(after):
        raise ValueError("private artifact changed through retained file object")
    return _PinnedFile(
        path=pin.path,
        fingerprint=before_fingerprint,
        bytes=byte_count,
        sha256=digest.hexdigest(),
    )


def _open_inspected_archive(
    pin: _PinnedFile,
    expected_members: list[dict[str, Any]],
) -> _OpenPinnedArchive:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    source: BinaryIO | None = None
    archive: zipfile.ZipFile | None = None
    try:
        descriptor = os.open(pin.path, flags)
        source = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        opened = _digest_open_file(source, pin)
        if opened != pin:
            raise ValueError("training archive changed before retained inspection")
        archive = zipfile.ZipFile(source, mode="r")
        digest = inspect_open_training_archive(
            archive,
            expected_members,
            name=pin.path.name,
        )
        if digest.bytes != pin.bytes or digest.sha256 != pin.sha256:
            raise ValueError("training archive changed during inspection")
        if _digest_open_file(source, pin) != pin:
            raise ValueError("training archive changed during inspection")
        return _OpenPinnedArchive(
            source=source,
            archive=archive,
            pin=pin,
            digest=digest,
        )
    except Exception:
        if archive is not None:
            archive.close()
        if source is not None:
            source.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise


def _selected_holdout_paths(context: _PathContext, selection: dict[str, Any]) -> list[Path]:
    items = selection.get("items")
    if type(items) is not list:
        raise ValueError("provider selection items are invalid")
    paths: list[Path] = []
    for item in items:
        _exact_fields(item, PROVIDER_ITEM_FIELDS, label="provider selection item")
        for role in ("image", "audio"):
            record = _exact_fields(item[role], PROVIDER_DIGEST_FIELDS, label="provider selection digest")
            name = record.get("name")
            if type(name) is not str:
                raise ValueError("provider selection filename is invalid")
            safe_relative_name(name)
            if "/" in name or "\\" in name:
                raise ValueError("provider selection filename is invalid")
            paths.append(context.run_dir / "candidates" / name)
    return paths


def _pin_root_artifacts(
    context: _PathContext,
    artifacts: _LoadedArtifacts,
    require_receipt: bool,
) -> dict[str, _PinnedFile]:
    root_pin = _pin_file(context.run_dir / "bundle" / "bundle-manifest.json")
    root_content = canonical_json_bytes(artifacts.root_manifest)
    if (
        root_pin.bytes != len(root_content)
        or root_pin.sha256 != hashlib.sha256(root_content).hexdigest()
    ):
        raise ValueError("root manifest changed after canonical parsing")
    pins: dict[str, _PinnedFile] = {"root_manifest": root_pin}
    parsed_roles = {
        "standing_authorization": artifacts.standing_policy,
        "price_evidence": artifacts.price_evidence,
        "structural_report": artifacts.structural_report,
        "quality_attestation": artifacts.quality_attestation,
        "dataset_manifest": artifacts.dataset_manifest,
        "execution_config": artifacts.execution_config,
        "provider_validation_selection": artifacts.provider_selection,
    }
    root_records = artifacts.root_manifest.get("artifacts")
    if type(root_records) is not dict or set(root_records) != REQUIRED_ROOT_ARTIFACT_ROLES:
        raise ValueError("root artifact roles are invalid")
    for role, relative in ROOT_ARTIFACT_PATHS.items():
        pin = _pin_file(context.run_dir / relative)
        record = root_records.get(role)
        if (
            type(record) is not dict
            or set(record) != {"bytes", "sha256"}
            or record.get("bytes") != pin.bytes
            or record.get("sha256") != pin.sha256
        ):
            raise ValueError("root artifact digest mismatch")
        if role in parsed_roles:
            parsed_content = canonical_json_bytes(parsed_roles[role])
            if (
                pin.bytes != len(parsed_content)
                or pin.sha256
                != hashlib.sha256(parsed_content).hexdigest()
            ):
                raise ValueError("parsed root artifact changed before pinning")
        pins[role] = pin
    for index, path in enumerate(_selected_holdout_paths(context, artifacts.provider_selection)):
        pin = _pin_file(path)
        record = artifacts.provider_selection["items"][index // 2][
            "image" if index % 2 == 0 else "audio"
        ]
        if record.get("bytes") != pin.bytes or record.get("sha256") != pin.sha256:
            raise ValueError("selected holdout digest mismatch")
        pins[f"selected_{index}"] = pin
    if require_receipt:
        receipt_path = context.run_dir / "control" / "execution-approval.json"
        if artifacts.receipt is not None:
            receipt_pin = _pin_file(receipt_path)
            receipt_content = canonical_json_bytes(artifacts.receipt)
            if (
                receipt_pin.bytes != len(receipt_content)
                or receipt_pin.sha256 != hashlib.sha256(receipt_content).hexdigest()
            ):
                raise ValueError("execution receipt changed after canonical parsing")
            pins["receipt"] = receipt_pin
    identities = [pin.fingerprint for pin in pins.values()]
    comparable = [(item.device, item.inode) for item in identities]
    if len(set(comparable)) != len(comparable):
        raise ValueError("private artifacts must not alias")
    return pins


def _expected_archive_members(dataset_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    members = dataset_manifest.get("training_members")
    if type(members) is not list:
        raise ValueError("dataset training members are invalid")
    return members


def _spec_from_report(report: Mapping[str, Any], *, minimum: int) -> A2VSpec:
    spec = report.get("spec")
    if type(spec) is not dict:
        raise ValueError("structural spec is invalid")
    try:
        return A2VSpec(
            width=spec["width"],
            height=spec["height"],
            frames=spec["frames"],
            fps=spec["fps"],
            sample_rate=spec["sample_rate"],
            min_groups=minimum,
        )
    except (KeyError, TypeError):
        raise ValueError("structural spec is invalid") from None


def _windows_extended_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


def _safe_archive_structural_validation(
    context: _PathContext,
    artifacts: _LoadedArtifacts,
    opened_archive: _OpenPinnedArchive,
    *,
    temporary_parent: Path | None = None,
) -> dict[str, Any]:
    run = context.run_dir
    parent = run / ".preflight-tmp" if temporary_parent is None else Path(temporary_parent)
    parent_created = False
    temporary: Path | None = None
    try:
        if parent.exists() or parent.is_symlink():
            metadata = _protected_metadata(parent)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("preflight temporary parent is invalid")
            if temporary_parent is not None and any(parent.iterdir()):
                raise ValueError("preflight temporary parent is not empty")
        else:
            if temporary_parent is not None:
                raise ValueError("preflight temporary parent is unavailable")
            parent.mkdir(mode=0o700)
            parent_created = True
            if os.name != "nt":
                os.chmod(parent, 0o700)
        temporary = Path(tempfile.mkdtemp(prefix="archive-", dir=parent))
        if os.name != "nt":
            os.chmod(temporary, 0o700)
        expected = {record["name"]: record for record in _expected_archive_members(artifacts.dataset_manifest)}
        archive = opened_archive.archive
        for info in archive.infolist():
            name = info.filename
            safe_relative_name(name)
            if "/" in name or "\\" in name or name not in expected:
                raise ValueError("archive member name is unsafe")
            destination = temporary / name
            opened_destination = _windows_extended_path(destination)
            digest = hashlib.sha256()
            byte_count = 0
            with archive.open(info, mode="r") as source, opened_destination.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if os.name != "nt":
                os.chmod(opened_destination, 0o600)
            record = expected[name]
            if byte_count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("extracted archive member digest mismatch")
        train_groups = artifacts.dataset_manifest.get("groups", {}).get("train")
        if type(train_groups) is not list:
            raise ValueError("dataset training groups are invalid")
        fresh = validate_a2v_directory(
            _windows_extended_path(temporary),
            spec=_spec_from_report(artifacts.structural_report, minimum=len(train_groups)),
            trigger_phrase=artifacts.execution_config.get("trigger_phrase"),
        )
        expected_report = {
            "schema_version": artifacts.structural_report["schema_version"],
            "status": "valid",
            "spec": artifacts.structural_report["spec"],
            "groups": train_groups,
        }
        if canonical_json_bytes(fresh) != canonical_json_bytes(expected_report):
            raise ValueError("extracted archive structural report mismatch")
        if _digest_open_file(opened_archive.source, opened_archive.pin) != opened_archive.pin:
            raise ValueError("training archive changed during structural validation")
        return fresh
    finally:
        if temporary is not None:
            shutil.rmtree(_windows_extended_path(temporary), ignore_errors=False)
        if parent_created:
            parent.rmdir()


def _create_static_archive_parent(context: _PathContext) -> tuple[Path, _NodeFingerprint]:
    try:
        parent = Path(tempfile.mkdtemp(prefix=".a2v-static-", dir=context.private_root))
    except OSError as exc:
        raise ValueError("static archive temporary parent is unavailable") from exc
    try:
        metadata = _protected_metadata(parent)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("static archive temporary parent is invalid")
        return parent, _node_fingerprint(metadata)
    except Exception:
        try:
            parent.rmdir()
        except OSError:
            pass
        raise


def _remove_static_archive_parent(parent: Path, fingerprint: _NodeFingerprint) -> None:
    try:
        metadata = _protected_metadata(parent)
    except Exception:
        raise ValueError("static archive temporary parent changed") from None
    if _node_fingerprint(metadata) != fingerprint:
        raise ValueError("static archive temporary parent changed")
    try:
        if any(parent.iterdir()):
            raise ValueError("static archive temporary parent changed")
        parent.rmdir()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("static archive temporary parent changed") from exc


def _candidate_structural_rerun(
    context: _PathContext,
    artifacts: _LoadedArtifacts,
) -> dict[str, Any]:
    fresh = validate_a2v_directory(
        context.run_dir / "candidates",
        spec=_spec_from_report(artifacts.structural_report, minimum=10),
        trigger_phrase=artifacts.execution_config.get("trigger_phrase"),
    )
    if canonical_json_bytes(fresh) != canonical_json_bytes(artifacts.structural_report):
        raise ValueError("candidate structural report mismatch")
    return fresh


def _validate_request_allowlist(
    context: _PathContext,
    artifacts: _LoadedArtifacts,
    pins: dict[str, _PinnedFile],
) -> dict[str, Any]:
    config = validate_execution_config(artifacts.execution_config)
    root = artifacts.root_manifest
    expected = {
        "execution_id": context.execution_id,
        "pilot_id": context.pilot_id,
        "created_at_utc": root.get("created_at_utc"),
        "expires_at_utc": root.get("expires_at_utc"),
        "dataset_manifest_sha256": pins["dataset_manifest"].sha256,
        "training_archive_sha256": pins["training_archive"].sha256,
        "standing_authorization_sha256": pins["standing_authorization"].sha256,
        "price_evidence_sha256": pins["price_evidence"].sha256,
    }
    for field_name, value in expected.items():
        if config.get(field_name) != value:
            raise ValueError("execution configuration root binding mismatch")
    if root.get("execution_id") != context.execution_id:
        raise ValueError("root execution identity mismatch")
    return config


def _verify_static_gates(
    context: _PathContext,
    expected_bundle_id: str,
    *,
    require_receipt: bool,
    archive_temporary_parent: Path | None = None,
) -> _StaticVerification:
    """Run the integrity-only prefix of preflight in its public gate order."""

    passed: list[str] = []
    opened_archive: _OpenPinnedArchive | None = None
    try:
        try:
            artifacts = _load_canonical_artifacts(context, require_receipt)
        except Exception:
            raise _StaticGateFailure("canonical_artifacts", tuple(passed)) from None
        passed.append("canonical_artifacts")

        try:
            if SHA256_PATTERN.fullmatch(expected_bundle_id) is None:
                raise ValueError("confirmed bundle ID is invalid")
            bundle_id = compute_bundle_id(artifacts.root_manifest)
            if bundle_id != expected_bundle_id:
                raise ValueError("confirmed bundle ID mismatch")
        except Exception:
            raise _StaticGateFailure("bundle_id", tuple(passed)) from None
        passed.append("bundle_id")

        try:
            pins = _pin_root_artifacts(context, artifacts, require_receipt)
        except Exception:
            raise _StaticGateFailure("root_artifact_hashes", tuple(passed)) from None
        passed.append("root_artifact_hashes")

        try:
            opened_archive = _open_inspected_archive(
                pins["training_archive"],
                _expected_archive_members(artifacts.dataset_manifest),
            )
            archive_digest = opened_archive.digest
        except Exception:
            raise _StaticGateFailure("archive_inspection", tuple(passed)) from None
        passed.append("archive_inspection")

        try:
            _safe_archive_structural_validation(
                context,
                artifacts,
                opened_archive,
                temporary_parent=archive_temporary_parent,
            )
        except Exception:
            raise _StaticGateFailure("archive_structural_validation", tuple(passed)) from None
        passed.append("archive_structural_validation")

        try:
            fresh_structural = _candidate_structural_rerun(context, artifacts)
        except Exception:
            raise _StaticGateFailure("candidate_structural_rerun", tuple(passed)) from None
        passed.append("candidate_structural_rerun")

        try:
            quality_summary = validate_quality_and_splits(
                artifacts.quality_attestation,
                fresh_structural,
            )
        except Exception:
            raise _StaticGateFailure("quality_attestation", tuple(passed)) from None
        passed.append("quality_attestation")

        try:
            rebuilt_manifest = build_dataset_manifest(
                fresh_structural,
                artifacts.quality_attestation,
                FileDigest(
                    name=pins["training_archive"].path.name,
                    bytes=archive_digest.bytes,
                    sha256=archive_digest.sha256,
                ),
                candidate_dir=context.run_dir / "candidates",
            )
            if canonical_json_bytes(rebuilt_manifest) != canonical_json_bytes(
                artifacts.dataset_manifest
            ):
                raise ValueError("dataset manifest is stale")
            if artifacts.root_manifest.get("holdout_groups") != rebuilt_manifest["groups"]["holdout"]:
                raise ValueError("root holdout manifest mismatch")
        except Exception:
            raise _StaticGateFailure("split_and_manifest", tuple(passed)) from None
        passed.append("split_and_manifest")

        try:
            try:
                validate_execution_config(artifacts.execution_config)
            except Exception:
                # Preserve the legacy gate attribution: malformed request fields
                # are owned by the following request-allowlist gate.
                items = artifacts.provider_selection.get("items")
                if type(items) is not list:
                    raise ValueError("provider validation items are invalid")
                provider_items = len(items)
                provider_selection = copy.deepcopy(artifacts.provider_selection)
            else:
                provider_selection = validate_provider_validation_selection(
                    artifacts.provider_selection,
                    fresh_structural,
                    quality_summary,
                    artifacts.execution_config,
                    context.run_dir / "candidates",
                )
                provider_items = len(provider_selection["items"])
        except Exception:
            raise _StaticGateFailure("provider_validation_selection", tuple(passed)) from None
        passed.append("provider_validation_selection")

        try:
            config = _validate_request_allowlist(context, artifacts, pins)
        except Exception:
            raise _StaticGateFailure("request_allowlist", tuple(passed)) from None
        passed.append("request_allowlist")

        return _StaticVerification(
            context=context,
            artifacts=artifacts,
            pins=pins,
            bundle_id=bundle_id,
            archive_digest=archive_digest,
            fresh_structural=fresh_structural,
            quality_summary=quality_summary,
            rebuilt_manifest=rebuilt_manifest,
            provider_selection=provider_selection,
            config=config,
            provider_items=provider_items,
            passed_gates=tuple(passed),
        )
    finally:
        if opened_archive is not None:
            opened_archive.close()


def _public_static_bundle(result: _StaticVerification) -> StaticA2VBundle:
    return StaticA2VBundle(
        private_root=result.context.private_root,
        run_dir=result.context.run_dir,
        pilot_id=result.context.pilot_id,
        execution_id=result.context.execution_id,
        bundle_id=result.bundle_id,
        root_manifest=copy.deepcopy(result.artifacts.root_manifest),
        structural_report=copy.deepcopy(result.fresh_structural),
        quality_attestation=copy.deepcopy(result.artifacts.quality_attestation),
        dataset_manifest=copy.deepcopy(result.rebuilt_manifest),
        provider_selection=copy.deepcopy(result.provider_selection),
        execution_config=copy.deepcopy(result.config),
        price_evidence=copy.deepcopy(result.artifacts.price_evidence),
        standing_policy=copy.deepcopy(result.artifacts.standing_policy),
        quality_summary=copy.deepcopy(result.quality_summary),
        archive_digest=FileDigest(
            name=result.archive_digest.name,
            bytes=result.archive_digest.bytes,
            sha256=result.archive_digest.sha256,
        ),
        passed_gates=result.passed_gates,
    )


def verify_static_a2v_bundle(
    private_root: Path,
    run_dir: Path,
    expected_bundle_id: str,
) -> StaticA2VBundle:
    """Verify static source provenance without treating it as live authority."""

    try:
        context = _lexical_context(Path(private_root), Path(run_dir))
    except Exception:
        raise ValueError("canonical private source run is required") from None
    temporary_parent, fingerprint = _create_static_archive_parent(context)
    try:
        try:
            result = _verify_static_gates(
                context,
                expected_bundle_id,
                require_receipt=False,
                archive_temporary_parent=temporary_parent,
            )
        except _StaticGateFailure as exc:
            raise ValueError(f"static A2V bundle failed at {exc.gate}") from None
        return _public_static_bundle(result)
    finally:
        _remove_static_archive_parent(temporary_parent, fingerprint)
