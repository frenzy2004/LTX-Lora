from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import re
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
from .a2v_dataset import A2VSpec, STRUCTURAL_REPORT_SCHEMA, validate_a2v_directory
from .a2v_quality import (
    ATTESTATION_KEYS,
    QUALITY_ATTESTATION_SCHEMA,
    validate_quality_and_splits,
)
from .artifacts import (
    FileDigest,
    atomic_write_json,
    canonical_json_bytes,
    safe_relative_name,
    strict_load_json,
)
from .authorization import (
    APPROVAL_SCHEMA_VERSION,
    CUMULATIVE_CAP_USD,
    EXECUTION_CONFIG_FIELDS,
    EXECUTION_CONFIG_SCHEMA_VERSION,
    EXECUTION_RECEIPT_FIELDS,
    PRICE_EVIDENCE_FIELDS,
    STANDING_AUTHORIZATION_FIELDS,
    TRAINING_MAX_USD,
    ExecutionReceipt,
    PriceEvidence,
    StandingAuthorization,
    validate_execution_config,
    verify_execution_receipt,
)
from .pilot_ledger import (
    CAP_USD_TEXT,
    SQLITE_SIDECAR_SUFFIXES,
    LedgerPreflightSnapshot,
    PilotLedger,
)
from .private_workspace import (
    EXECUTION_ID_PATTERN,
    PILOT_ID_PATTERN,
    require_canonical_run_dir,
    resolve_pilot_ledger,
)
from .provider_validation import (
    CANONICAL_JSON_VERSION as PROVIDER_CANONICAL_JSON_VERSION,
    DIGEST_FIELDS as PROVIDER_DIGEST_FIELDS,
    ITEM_FIELDS as PROVIDER_ITEM_FIELDS,
    SCHEMA_VERSION as PROVIDER_SELECTION_SCHEMA,
    SELECTION_FIELDS as PROVIDER_SELECTION_FIELDS,
    validate_provider_validation_selection,
)


GATE_ORDER = (
    "private_root",
    "canonical_artifacts",
    "bundle_id",
    "root_artifact_hashes",
    "archive_inspection",
    "archive_structural_validation",
    "candidate_structural_rerun",
    "quality_attestation",
    "split_and_manifest",
    "provider_validation_selection",
    "request_allowlist",
    "price_freshness",
    "standing_policy",
    "receipt",
    "ledger_snapshot",
    "final_recheck",
)

PREFLIGHT_REPORT_SCHEMA = "a2v-preflight-report-v1"
TRAINING_RESERVATION_USD = TRAINING_MAX_USD
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
MONEY_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{4}", re.ASCII)
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


class PreflightNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class PreflightStatus:
    schema_version: str
    status: str
    failed_gate: str | None
    receipt_required: bool
    bundle_id: str
    execution_id: str | None
    training_groups: int | None
    holdout_groups: int | None
    provider_validation_items: int | None
    committed_usd: str | None
    remaining_usd: str | None
    training_reservation_usd: str
    remaining_after_reservation_usd: str | None
    passed_gates: tuple[str, ...]
    pilot_id: str | None = field(repr=False)
    ledger_id: str | None = field(repr=False)
    ledger_head_sha256: str | None = field(repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "failed_gate": self.failed_gate,
            "receipt_required": self.receipt_required,
            "bundle_id": self.bundle_id,
            "execution_id": self.execution_id,
            "counts": {
                "training_groups": self.training_groups,
                "holdout_groups": self.holdout_groups,
                "provider_validation_items": self.provider_validation_items,
            },
            "budget": {
                "committed_usd": self.committed_usd,
                "remaining_usd": self.remaining_usd,
                "training_reservation_usd": self.training_reservation_usd,
                "remaining_after_reservation_usd": self.remaining_after_reservation_usd,
            },
            "passed_gates": list(self.passed_gates),
        }

    def require_ready(self) -> PreflightStatus:
        if not (
            self.status == "ready_for_paid_execution"
            and self.receipt_required is True
            and self.failed_gate is None
            and self.ledger_head_sha256 is not None
            and self.passed_gates == GATE_ORDER
        ):
            raise PreflightNotReady("preflight is not ready for paid execution")
        return self


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
        self.archive.close()
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
) -> dict[str, Any]:
    run = context.run_dir
    parent = run / ".preflight-tmp"
    parent_created = False
    temporary: Path | None = None
    try:
        if parent.exists() or parent.is_symlink():
            metadata = _protected_metadata(parent)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("preflight temporary parent is invalid")
        else:
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


def _parse_expiry(value: Any) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("expiry timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("expiry timestamp is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError("expiry timestamp is invalid")
    return parsed


def _clock_read(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo != timezone.utc:
        raise ValueError("preflight clock must return UTC")
    return value.replace(microsecond=0)


def _money(value: Any) -> Decimal:
    if type(value) is not str or MONEY_PATTERN.fullmatch(value) is None:
        raise ValueError("ledger money is invalid")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("ledger money is invalid") from exc


def _ledger_sidecars_absent(path: Path) -> None:
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("ledger sidecar is prohibited")


def _ledger_snapshot(
    context: _PathContext,
    config: dict[str, Any],
    bundle_id: str,
    receipt: ExecutionReceipt | None,
) -> tuple[LedgerPreflightSnapshot, Path, str, _PinnedFile]:
    ledger_id = config.get("ledger_id")
    if type(ledger_id) is not str:
        raise ValueError("ledger identity is invalid")
    ledger_path = resolve_pilot_ledger(context.private_root, context.pilot_id)
    _ledger_sidecars_absent(ledger_path)
    ledger_pin = _pin_file(ledger_path)
    ledger = PilotLedger.open_existing(
        ledger_path,
        context.pilot_id,
        expected_ledger_id=ledger_id,
    )
    snapshot = ledger.preflight_snapshot(bundle_id, context.execution_id)
    expected = {
        "pilot_id": context.pilot_id,
        "ledger_id": ledger_id,
        "bundle_id": bundle_id,
        "execution_id": context.execution_id,
    }
    for name, value in expected.items():
        if getattr(snapshot, name) != value:
            raise ValueError("ledger snapshot identity mismatch")
    if type(snapshot.replay_detected) is not bool or snapshot.replay_detected:
        raise ValueError("ledger snapshot replay detected")
    committed = _money(snapshot.committed_usd)
    remaining = _money(snapshot.remaining_usd)
    if committed + remaining != Decimal(CAP_USD_TEXT):
        raise ValueError("ledger snapshot balance mismatch")
    if remaining < Decimal(TRAINING_RESERVATION_USD):
        raise ValueError("ledger has insufficient remaining budget")
    if SHA256_PATTERN.fullmatch(snapshot.head_sha256) is None:
        raise ValueError("ledger snapshot head is invalid")
    if receipt is not None and receipt.ledger_head_sha256 != snapshot.head_sha256:
        raise ValueError("execution receipt ledger head mismatch")
    _ledger_sidecars_absent(ledger_path)
    return (
        snapshot,
        ledger_path,
        f"{remaining - Decimal(TRAINING_RESERVATION_USD):.4f}",
        ledger_pin,
    )


def _final_recheck(
    *,
    context: _PathContext,
    artifacts: _LoadedArtifacts,
    pins: dict[str, _PinnedFile],
    clock: Callable[[], datetime],
    receipt: ExecutionReceipt | None,
    ledger_path: Path,
) -> None:
    current = _clock_read(clock)
    PriceEvidence.from_dict(artifacts.price_evidence, now=current)
    StandingAuthorization.from_dict(artifacts.standing_policy, now=current)
    if receipt is not None:
        ExecutionReceipt.from_dict(receipt.to_dict(), now=current)
    for value in (
        artifacts.root_manifest.get("expires_at_utc"),
        artifacts.execution_config.get("expires_at_utc"),
    ):
        if _parse_expiry(value) <= current:
            raise ValueError("preflight artifact expired")
    _ledger_sidecars_absent(ledger_path)
    if _security_snapshot(context.private_root) != context.security_snapshot:
        raise ValueError("private path identity changed during preflight")
    for pin in pins.values():
        if _pin_file(pin.path) != pin:
            raise ValueError("private artifact changed during preflight")
    _ledger_sidecars_absent(ledger_path)


def _atomic_report(context: _PathContext, report: PreflightStatus) -> None:
    path = context.run_dir / "control" / "preflight-report.json"
    atomic_write_json(path, report.to_public_dict())
    if os.name != "nt":
        os.chmod(path, 0o600)


def run_preflight(
    run_dir: Path,
    confirmed_bundle_id: str,
    *,
    require_receipt: bool,
    approved_private_root: Path,
    clock: Callable[[], datetime],
) -> PreflightStatus:
    if type(require_receipt) is not bool or not callable(clock):
        raise TypeError("preflight arguments are invalid")
    public_bundle_id = (
        confirmed_bundle_id
        if type(confirmed_bundle_id) is str and SHA256_PATTERN.fullmatch(confirmed_bundle_id)
        else ""
    )
    passed: list[str] = []
    context: _PathContext | None = None
    execution_id: str | None = None
    training_groups: int | None = None
    holdout_groups: int | None = None
    provider_items: int | None = None
    committed_usd: str | None = None
    remaining_usd: str | None = None
    remaining_after: str | None = None
    pilot_id: str | None = None
    ledger_id: str | None = None
    ledger_head: str | None = None
    opened_archive: _OpenPinnedArchive | None = None

    def finish(failed_gate: str | None) -> PreflightStatus:
        if opened_archive is not None:
            opened_archive.close()
        if failed_gate is None:
            status = (
                "ready_for_paid_execution"
                if require_receipt
                else "ready_for_policy_issuance"
            )
        else:
            status = "failed"
        report = PreflightStatus(
            schema_version=PREFLIGHT_REPORT_SCHEMA,
            status=status,
            failed_gate=failed_gate,
            receipt_required=require_receipt,
            bundle_id=public_bundle_id,
            execution_id=execution_id,
            training_groups=training_groups,
            holdout_groups=holdout_groups,
            provider_validation_items=provider_items,
            committed_usd=committed_usd,
            remaining_usd=remaining_usd,
            training_reservation_usd=TRAINING_RESERVATION_USD,
            remaining_after_reservation_usd=remaining_after,
            passed_gates=tuple(passed),
            pilot_id=pilot_id,
            ledger_id=ledger_id,
            ledger_head_sha256=ledger_head,
        )
        if context is not None:
            _atomic_report(context, report)
        return report

    try:
        context = _lexical_context(Path(approved_private_root), Path(run_dir))
        execution_id = context.execution_id
        pilot_id = context.pilot_id
    except Exception:
        return finish("private_root")
    passed.append("private_root")

    try:
        artifacts = _load_canonical_artifacts(context, require_receipt)
    except Exception:
        return finish("canonical_artifacts")
    passed.append("canonical_artifacts")

    try:
        if SHA256_PATTERN.fullmatch(public_bundle_id) is None:
            raise ValueError("confirmed bundle ID is invalid")
        computed_bundle_id = compute_bundle_id(artifacts.root_manifest)
        if computed_bundle_id != public_bundle_id:
            raise ValueError("confirmed bundle ID mismatch")
    except Exception:
        return finish("bundle_id")
    passed.append("bundle_id")

    try:
        pins = _pin_root_artifacts(context, artifacts, require_receipt)
    except Exception:
        return finish("root_artifact_hashes")
    passed.append("root_artifact_hashes")

    try:
        opened_archive = _open_inspected_archive(
            pins["training_archive"],
            _expected_archive_members(artifacts.dataset_manifest),
        )
        archive_digest = opened_archive.digest
    except Exception:
        return finish("archive_inspection")
    passed.append("archive_inspection")

    try:
        _safe_archive_structural_validation(
            context,
            artifacts,
            opened_archive,
        )
    except Exception:
        return finish("archive_structural_validation")
    passed.append("archive_structural_validation")

    try:
        fresh_structural = _candidate_structural_rerun(context, artifacts)
    except Exception:
        return finish("candidate_structural_rerun")
    passed.append("candidate_structural_rerun")

    try:
        quality_summary = validate_quality_and_splits(
            artifacts.quality_attestation,
            fresh_structural,
        )
        training_groups = quality_summary["coverage_counts"]["accepted_train_groups"]
        holdout_groups = quality_summary["coverage_counts"]["accepted_holdout_groups"]
    except Exception:
        return finish("quality_attestation")
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
        if canonical_json_bytes(rebuilt_manifest) != canonical_json_bytes(artifacts.dataset_manifest):
            raise ValueError("dataset manifest is stale")
        if artifacts.root_manifest.get("holdout_groups") != rebuilt_manifest["groups"]["holdout"]:
            raise ValueError("root holdout manifest mismatch")
    except Exception:
        return finish("split_and_manifest")
    passed.append("split_and_manifest")

    try:
        try:
            validate_execution_config(artifacts.execution_config)
        except Exception:
            # The selection validator intentionally consumes the exact request
            # validator. Preserve the owning gate for a malformed request; the
            # paid boundary remains closed at the immediately following gate.
            items = artifacts.provider_selection.get("items")
            if type(items) is not list:
                raise ValueError("provider validation items are invalid")
            provider_items = len(items)
        else:
            validated_selection = validate_provider_validation_selection(
                artifacts.provider_selection,
                fresh_structural,
                quality_summary,
                artifacts.execution_config,
                context.run_dir / "candidates",
            )
            provider_items = len(validated_selection["items"])
    except Exception:
        return finish("provider_validation_selection")
    passed.append("provider_validation_selection")

    try:
        config = _validate_request_allowlist(context, artifacts, pins)
        ledger_id = config["ledger_id"]
    except Exception:
        return finish("request_allowlist")
    passed.append("request_allowlist")

    try:
        price = PriceEvidence.from_dict(artifacts.price_evidence, now=_clock_read(clock))
    except Exception:
        return finish("price_freshness")
    passed.append("price_freshness")

    try:
        policy = StandingAuthorization.from_dict(
            artifacts.standing_policy,
            now=_clock_read(clock),
        )
        if config["endpoint"] != policy.endpoint or config["steps"] != policy.steps:
            raise ValueError("standing policy request mismatch")
        if (
            config["training_max_usd"] != policy.training_max_usd
            or config["validation_allocation_usd"] != policy.validation_allocation_usd
            or config["cumulative_cap_usd"] != policy.cumulative_cap_usd
            or config["cumulative_cap_usd"] != CUMULATIVE_CAP_USD
        ):
            raise ValueError("standing policy cost mismatch")
        root_expiry = _parse_expiry(artifacts.root_manifest["expires_at_utc"])
        if root_expiry > _parse_expiry(policy.expires_at_utc) or root_expiry > _parse_expiry(price.expires_at_utc):
            raise ValueError("bundle expiry exceeds authorization evidence")
    except Exception:
        return finish("standing_policy")
    passed.append("standing_policy")

    try:
        receipt: ExecutionReceipt | None = None
        if require_receipt:
            if artifacts.receipt is None:
                raise ValueError("execution receipt is required")
            receipt = verify_execution_receipt(
                artifacts.receipt,
                policy,
                context.run_dir,
                now=_clock_read(clock),
            )
    except Exception:
        return finish("receipt")
    passed.append("receipt")

    try:
        snapshot, ledger_path, remaining_after, ledger_pin = _ledger_snapshot(
            context,
            config,
            public_bundle_id,
            receipt,
        )
        pins["ledger"] = ledger_pin
        committed_usd = snapshot.committed_usd
        remaining_usd = snapshot.remaining_usd
        ledger_head = snapshot.head_sha256
    except Exception:
        return finish("ledger_snapshot")
    passed.append("ledger_snapshot")

    try:
        _final_recheck(
            context=context,
            artifacts=artifacts,
            pins=pins,
            clock=clock,
            receipt=receipt,
            ledger_path=ledger_path,
        )
    except Exception:
        return finish("final_recheck")
    passed.append("final_recheck")
    return finish(None)
