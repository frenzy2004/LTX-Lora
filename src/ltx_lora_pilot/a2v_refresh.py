from __future__ import annotations

import copy
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterator, Mapping

from .a2v_bundle import (
    build_dataset_manifest,
    build_root_manifest,
    build_training_archive,
    compute_bundle_id,
    inspect_training_archive,
)
from .a2v_dataset import A2VSpec, validate_a2v_directory
from .a2v_quality import validate_quality_and_splits
from .artifacts import canonical_json_bytes, sha256_file, strict_load_json
from .authorization import PriceEvidence, StandingAuthorization, validate_execution_config
from . import preflight as _preflight
from .preflight import verify_static_a2v_bundle
from .private_workspace import (
    canonical_new_run_dir,
    require_canonical_private_file,
    require_canonical_run_dir,
)
from .provider_validation import (
    build_provider_validation_selection,
    validate_provider_validation_selection,
)
from .staging import _copy_sealed_file


@dataclass(frozen=True)
class SourceRunSnapshot:
    run_dir: Path
    structural_report: dict[str, Any]
    quality_attestation: dict[str, Any]
    quality_summary: dict[str, Any]
    source_config: dict[str, Any]
    private_root: Path | None = None
    pilot_id: str | None = None
    execution_id: str | None = None
    bundle_id: str | None = None


def _require_dataset_contract(
    run_dir: Path,
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
) -> set[str]:
    groups = structural_report.get("groups")
    train_ids = quality_summary.get("accepted_train_group_ids")
    holdout_ids = quality_summary.get("accepted_holdout_group_ids")
    if (
        type(groups) is not list
        or type(train_ids) is not list
        or type(holdout_ids) is not list
        or len(groups) != 17
        or len(train_ids) != 12
        or len(holdout_ids) != 5
    ):
        raise ValueError("source dataset must contain exactly 17 accepted groups")
    group_ids = [group.get("group_id") for group in groups if type(group) is dict]
    if len(group_ids) != 17 or len(set(group_ids)) != 17:
        raise ValueError("source structural groups are invalid")
    if set(train_ids) & set(holdout_ids) or set(train_ids) | set(holdout_ids) != set(group_ids):
        raise ValueError("source split must cover the exact structural group set")
    expected_names: set[str] = set()
    for group in groups:
        if type(group) is not dict or type(group.get("files")) is not list:
            raise ValueError("source structural groups are invalid")
        files = group["files"]
        if len(files) != 4:
            raise ValueError("source groups must contain exactly four files")
        for record in files:
            if type(record) is not dict or type(record.get("name")) is not str:
                raise ValueError("source structural files are invalid")
            expected_names.add(record["name"])
    if len(expected_names) != 68:
        raise ValueError("source dataset must contain exactly 68 candidate files")
    candidate_dir = Path(run_dir) / "candidates"
    try:
        actual_names = {path.name for path in candidate_dir.iterdir()}
    except OSError as exc:
        raise ValueError("source candidate directory is unavailable") from exc
    if actual_names != expected_names or len(actual_names) != 68:
        raise ValueError("source candidate directory does not match the sealed groups")
    return expected_names


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_junction = getattr(os.path, "isjunction", None)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or bool(is_junction is not None and is_junction(path))
    )


def _has_ads_syntax(path: Path) -> bool:
    parts = path.parts[1:] if path.anchor else path.parts
    return any(":" in part for part in parts)


def _has_case_alias(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        try:
            with os.scandir(current) as entries:
                matching = {
                    entry.name
                    for entry in entries
                    if entry.name.casefold() == part.casefold()
                }
        except OSError:
            return True
        if matching != {part}:
            return True
        current /= part
    return False


def _has_alias_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _is_reparse_or_link(current):
            return True
    return False


def _require_canonical_source_run(run_dir: Path) -> Path:
    candidate = Path(run_dir)
    raw = str(candidate)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or _has_ads_syntax(candidate)
    ):
        raise ValueError("source run directory is invalid")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise ValueError("source run directory is invalid")
    if _has_alias_component(absolute) or _has_case_alias(absolute):
        raise ValueError("source run directory must be canonical")
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError("source run directory is unavailable") from exc
    if (
        _is_reparse_or_link(absolute)
        or not stat.S_ISDIR(metadata.st_mode)
        or str(resolved) != str(absolute)
    ):
        raise ValueError("source run directory must be canonical")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(parent).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _nearest_existing_ancestor(path: Path) -> Path:
    current = Path(path)
    while not current.exists() and not current.is_symlink():
        parent = current.parent
        if parent == current:
            raise ValueError("candidate staging destination is invalid")
        current = parent
    return current


def _prepare_empty_destination(destination: Path, *, prohibited_root: Path) -> Path:
    candidate = Path(destination)
    raw = str(candidate)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or _has_ads_syntax(candidate)
    ):
        raise ValueError("candidate staging destination is invalid")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise ValueError("candidate staging destination is invalid")
    if _is_within(absolute, prohibited_root):
        raise ValueError("candidate staging destination must not be within the source run")
    existing_parent = _nearest_existing_ancestor(absolute.parent)
    if _has_alias_component(existing_parent) or _has_case_alias(existing_parent):
        raise ValueError("candidate staging destination is invalid")
    try:
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if absolute.exists() or absolute.is_symlink():
            metadata = absolute.lstat()
            if _is_reparse_or_link(absolute) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("candidate staging destination is invalid")
            if any(absolute.iterdir()):
                raise ValueError("candidate staging destination must be empty")
        else:
            absolute.mkdir(mode=0o700)
            if os.name != "nt":
                os.chmod(absolute, 0o700)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("candidate staging destination is invalid") from exc
    if _has_alias_component(absolute) or _has_case_alias(absolute):
        raise ValueError("candidate staging destination is invalid")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError("candidate staging destination is invalid") from exc
    if str(resolved) != str(absolute):
        raise ValueError("candidate staging destination is invalid")
    if _is_within(resolved, prohibited_root):
        raise ValueError("candidate staging destination must not be within the source run")
    return resolved


def _snapshot_copy_inputs(
    snapshot: SourceRunSnapshot,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    if not isinstance(snapshot, SourceRunSnapshot):
        raise ValueError("source snapshot is invalid")
    structural = copy.deepcopy(snapshot.structural_report)
    attestation = copy.deepcopy(snapshot.quality_attestation)
    quality_summary = validate_quality_and_splits(attestation, structural)
    if canonical_json_bytes(quality_summary) != canonical_json_bytes(snapshot.quality_summary):
        raise ValueError("source quality split changed after verification")
    source_config = validate_execution_config(copy.deepcopy(snapshot.source_config))
    expected_names = _require_dataset_contract(
        snapshot.run_dir,
        structural,
        quality_summary,
    )
    return structural, attestation, source_config, expected_names


def _reverify_snapshot_for_copy(snapshot: SourceRunSnapshot) -> SourceRunSnapshot:
    """Reject caller-provided snapshots that no longer match sealed source state."""

    if not isinstance(snapshot, SourceRunSnapshot):
        raise ValueError("source snapshot is invalid")
    if (
        not isinstance(snapshot.private_root, Path)
        or type(snapshot.pilot_id) is not str
        or type(snapshot.execution_id) is not str
        or type(snapshot.bundle_id) is not str
        or not isinstance(snapshot.run_dir, Path)
    ):
        raise ValueError("source snapshot lacks verified source context")
    try:
        supplied_run = _require_canonical_source_run(snapshot.run_dir)
        refreshed = verify_source_run_static(
            private_root=snapshot.private_root,
            pilot_id=snapshot.pilot_id,
            source_execution_id=snapshot.execution_id,
            expected_source_bundle_id=snapshot.bundle_id,
        )
        if (
            str(supplied_run) != str(refreshed.run_dir)
            or str(snapshot.private_root) != str(refreshed.private_root)
            or snapshot.pilot_id != refreshed.pilot_id
            or snapshot.execution_id != refreshed.execution_id
            or snapshot.bundle_id != refreshed.bundle_id
        ):
            raise ValueError("source snapshot identity changed")
        for field_name in (
            "structural_report",
            "quality_attestation",
            "quality_summary",
            "source_config",
        ):
            if canonical_json_bytes(getattr(snapshot, field_name)) != canonical_json_bytes(
                getattr(refreshed, field_name)
            ):
                raise ValueError("source snapshot artifacts changed")
    except ValueError:
        raise
    except Exception:
        raise ValueError("source snapshot static verification failed") from None
    return refreshed


def verify_source_run_static(
    *,
    private_root: Path,
    pilot_id: str,
    source_execution_id: str,
    expected_source_bundle_id: str,
) -> SourceRunSnapshot:
    """Return a detached, static snapshot of the accepted source candidate set."""

    root = Path(private_root)
    run_dir = require_canonical_run_dir(
        root,
        pilot_id,
        source_execution_id,
        root / "pilots" / pilot_id / "runs" / source_execution_id,
    )
    bundle = verify_static_a2v_bundle(root, run_dir, expected_source_bundle_id)
    _require_dataset_contract(
        bundle.run_dir,
        bundle.structural_report,
        bundle.quality_summary,
    )
    return SourceRunSnapshot(
        run_dir=bundle.run_dir,
        structural_report=copy.deepcopy(bundle.structural_report),
        quality_attestation=copy.deepcopy(bundle.quality_attestation),
        quality_summary=copy.deepcopy(bundle.quality_summary),
        source_config=copy.deepcopy(bundle.execution_config),
        private_root=bundle.private_root,
        pilot_id=bundle.pilot_id,
        execution_id=bundle.execution_id,
        bundle_id=bundle.bundle_id,
    )


def copy_accepted_candidates(
    snapshot: SourceRunSnapshot,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Securely copy the exact accepted candidate set into an empty staging directory."""

    if not isinstance(snapshot, SourceRunSnapshot):
        raise ValueError("source snapshot is invalid")
    refreshed = _reverify_snapshot_for_copy(snapshot)
    source_run = _require_canonical_source_run(refreshed.run_dir)
    structural, attestation, source_config, expected_names = _snapshot_copy_inputs(refreshed)
    target = _prepare_empty_destination(destination, prohibited_root=source_run)
    source_dir = source_run / "candidates"
    if _is_reparse_or_link(source_dir) or not source_dir.is_dir():
        raise ValueError("source candidate directory is invalid")
    try:
        source_names = {path.name for path in source_dir.iterdir()}
    except OSError as exc:
        raise ValueError("source candidate directory is unavailable") from exc
    if source_names != expected_names or len(source_names) != 68:
        raise ValueError("source candidate directory does not match the accepted set")

    for group in structural["groups"]:
        for record in group["files"]:
            name = record["name"]
            _copy_sealed_file(
                source_dir / name,
                target / name,
                record,
                label="accepted source candidate",
            )

    try:
        copied_entries = list(target.iterdir())
    except OSError as exc:
        raise ValueError("copied candidate directory is unavailable") from exc
    if {path.name for path in copied_entries} != expected_names or len(copied_entries) != 68:
        raise ValueError("copied candidate directory does not match the accepted set")
    for path in copied_entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError("copied candidate is unavailable") from exc
        if (
            _is_reparse_or_link(path)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError("copied candidate is not an independent regular file")

    copied_structural = validate_a2v_directory(
        target,
        spec=A2VSpec(min_groups=17),
        trigger_phrase=source_config["trigger_phrase"],
    )
    if canonical_json_bytes(copied_structural) != canonical_json_bytes(structural):
        raise ValueError("copied candidate structural report changed")
    copied_summary = validate_quality_and_splits(attestation, copied_structural)
    if canonical_json_bytes(copied_summary) != canonical_json_bytes(refreshed.quality_summary):
        raise ValueError("copied candidate quality split changed")
    return copy.deepcopy(copied_structural), copy.deepcopy(attestation)


@dataclass(frozen=True)
class FreshA2VRunResult:
    execution_id: str
    bundle_id: str
    run_dir: Path


@dataclass(frozen=True)
class _StagingIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _PrivateFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class _FreshControlSnapshot:
    source: Path
    ancestors: tuple[tuple[Path, _StagingIdentity], ...]
    identity: _PrivateFileIdentity


@dataclass(frozen=True)
class _TrackedStagingFile:
    path: Path
    identity: _PrivateFileIdentity


@dataclass
class _TrackedStaging:
    path: Path
    parent: Path
    parent_identity: _StagingIdentity
    identity: _StagingIdentity
    directories: dict[Path, _StagingIdentity] = field(default_factory=dict)
    published: bool = False


_MOVEFILE_WRITE_THROUGH = 0x00000008
_STAGING_PREFIX = ".a2v-refresh-"
_ROOT_LAYOUT = frozenset({"candidates", "control", "validation", "bundle", "plan.md"})
_CONTROL_LAYOUT = frozenset(
    {
        "standing-authorization.json",
        "price-evidence.json",
        "structural-report.json",
        "quality-attestation.json",
        "execution-config.json",
    }
)
_VALIDATION_LAYOUT = frozenset({"provider-validation-selection.json"})
_BUNDLE_LAYOUT = frozenset(
    {"training-data.zip", "dataset-manifest.json", "bundle-manifest.json"}
)


def _staging_identity(path: Path) -> _StagingIdentity:
    try:
        metadata = Path(path).lstat()
    except OSError as exc:
        raise ValueError("staging path is unavailable") from exc
    if _is_reparse_or_link(Path(path)) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("staging path is invalid")
    return _StagingIdentity(int(metadata.st_dev), int(metadata.st_ino))


def _require_windows_explicit_owner_only_dacl(path: Path) -> None:
    """Reject an explicit DACL trustee outside the private Windows allowlist."""

    if os.name != "nt":
        return

    class _TrusteeW(ctypes.Structure):
        pass

    _TrusteeW._fields_ = [
        ("multiple_trustee", ctypes.POINTER(_TrusteeW)),
        ("multiple_trustee_operation", wintypes.DWORD),
        ("trustee_form", wintypes.DWORD),
        ("trustee_type", wintypes.DWORD),
        ("name", ctypes.c_void_p),
    ]

    class _ExplicitAccessW(ctypes.Structure):
        _fields_ = [
            ("access_permissions", wintypes.DWORD),
            ("access_mode", wintypes.DWORD),
            ("inheritance", wintypes.DWORD),
            ("trustee", _TrusteeW),
        ]

    local_free: Any | None = None
    descriptor = ctypes.c_void_p()
    entries = ctypes.POINTER(_ExplicitAccessW)()
    trusted_sids: list[ctypes.c_void_p] = []
    try:
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
        get_explicit_entries = advapi32.GetExplicitEntriesFromAclW
        get_explicit_entries.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.ULONG),
            ctypes.POINTER(ctypes.POINTER(_ExplicitAccessW)),
        ]
        get_explicit_entries.restype = wintypes.DWORD
        equal_sid = advapi32.EqualSid
        equal_sid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        equal_sid.restype = wintypes.BOOL
        convert_string_sid = advapi32.ConvertStringSidToSidW
        convert_string_sid.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        convert_string_sid.restype = wintypes.BOOL
        local_free = kernel32.LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p

        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        status = get_security(
            str(path),
            1,  # SE_FILE_OBJECT
            0x00000001 | 0x00000004,  # OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if status != 0 or not owner.value or not dacl.value or not descriptor.value:
            raise ValueError("private path DACL is unavailable")

        # CPython creates 0o700/0o600 Windows DACLs with these OS recovery
        # principals. They do not grant access to an arbitrary user: OWNER
        # RIGHTS resolves only to the owner, and SYSTEM/Administrators are
        # privileged local recovery identities.
        for sid_text in ("S-1-5-18", "S-1-5-32-544", "S-1-3-4"):
            sid = ctypes.c_void_p()
            if not convert_string_sid(sid_text, ctypes.byref(sid)) or not sid.value:
                raise ValueError("private path DACL is unavailable")
            trusted_sids.append(sid)

        entry_count = wintypes.ULONG()
        status = get_explicit_entries(
            dacl, ctypes.byref(entry_count), ctypes.byref(entries)
        )
        if status != 0 or (entry_count.value and not entries):
            raise ValueError("private path DACL is unavailable")
        for entry in entries[: entry_count.value]:
            trustee = entry.trustee
            if (
                trustee.multiple_trustee
                or trustee.trustee_form != 0  # TRUSTEE_IS_SID
                or not trustee.name
                or not (
                    equal_sid(owner, ctypes.c_void_p(trustee.name))
                    or any(
                        equal_sid(trusted, ctypes.c_void_p(trustee.name))
                        for trusted in trusted_sids
                    )
                )
            ):
                raise ValueError("private path DACL has an untrusted explicit ACE")
    except (AttributeError, OSError) as exc:
        raise ValueError("private path DACL is unavailable") from exc
    finally:
        if local_free is not None and entries:
            local_free(ctypes.cast(entries, ctypes.c_void_p))
        if local_free is not None:
            for sid in trusted_sids:
                local_free(sid)
        if local_free is not None and descriptor.value:
            local_free(descriptor)


def _require_staging_directory(path: Path) -> None:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ValueError("staging directory is unavailable") from exc
    if _is_reparse_or_link(candidate) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("staging directory is invalid")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("staging directory is not owner-only")
    try:
        _preflight._WINDOWS_DACL_CHECK(candidate)
        _require_windows_explicit_owner_only_dacl(candidate)
    except Exception:
        raise ValueError("staging directory is not private") from None


def _require_staging_regular_file(path: Path) -> os.stat_result:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ValueError("staging file is unavailable") from exc
    if (
        _is_reparse_or_link(candidate)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError("staging file is invalid")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("staging file is not owner-only")
    try:
        _preflight._WINDOWS_DACL_CHECK(candidate)
        _require_windows_explicit_owner_only_dacl(candidate)
    except Exception:
        raise ValueError("staging file is not private") from None
    return metadata


def _file_identity_from_metadata(metadata: os.stat_result) -> _PrivateFileIdentity:
    return _PrivateFileIdentity(
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _private_file_identity(path: Path) -> _PrivateFileIdentity:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ValueError("private control file is unavailable") from exc
    if (
        _is_reparse_or_link(candidate)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError("private control file is invalid")
    return _file_identity_from_metadata(metadata)


def _fresh_control_ancestor_paths(private_root: Path, source: Path) -> tuple[Path, ...]:
    root = Path(private_root)
    candidate = Path(source)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("private control file escaped its root") from exc
    if len(relative.parts) < 1:
        raise ValueError("private control file is invalid")
    paths = [root]
    current = root
    for part in relative.parts[:-1]:
        current /= part
        paths.append(current)
    return tuple(paths)


def _capture_fresh_control(private_root: Path, source: Path) -> _FreshControlSnapshot:
    canonical = require_canonical_private_file(private_root, source)
    ancestors = tuple(
        (path, _staging_identity(path))
        for path in _fresh_control_ancestor_paths(private_root, canonical)
    )
    return _FreshControlSnapshot(
        source=canonical,
        ancestors=ancestors,
        identity=_private_file_identity(canonical),
    )


def _verify_fresh_control_snapshot(
    private_root: Path, snapshot: _FreshControlSnapshot
) -> None:
    if not isinstance(snapshot, _FreshControlSnapshot):
        raise ValueError("private control snapshot is invalid")
    canonical = require_canonical_private_file(private_root, snapshot.source)
    if canonical != snapshot.source:
        raise ValueError("private control path changed")
    for directory, expected in snapshot.ancestors:
        _require_staging_directory(directory)
        if _staging_identity(directory) != expected:
            raise ValueError("private control parent changed")
    if _private_file_identity(snapshot.source) != snapshot.identity:
        raise ValueError("private control file changed")


class _WindowsPrivatePathGuard:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def close(self) -> None:
        if not self._handle:
            return
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = 0


def _open_windows_private_path_guard(
    path: Path, *, directory: bool
) -> _WindowsPrivatePathGuard:
    """Retain a no-write/no-delete handle while a private path is consumed."""

    if os.name != "nt":
        raise ValueError("Windows private path guard is unavailable")
    try:
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        flags = 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= 0x02000000  # FILE_FLAG_BACKUP_SEMANTICS
        handle = create_file(
            str(path),
            0x00000080,  # FILE_READ_ATTRIBUTES
            0x00000001,  # FILE_SHARE_READ only: deny write and delete sharing
            None,
            3,  # OPEN_EXISTING
            flags,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        return _WindowsPrivatePathGuard(int(handle))
    except (AttributeError, OSError) as exc:
        raise ValueError("private control path cannot be guarded") from exc


@contextmanager
def _pinned_fresh_controls(
    private_root: Path, snapshots: tuple[_FreshControlSnapshot, ...]
) -> Iterator[None]:
    """Freeze every control parent and leaf through all staged copies on Windows."""

    guards: list[_WindowsPrivatePathGuard] = []
    try:
        if os.name == "nt":
            ancestors: dict[Path, _StagingIdentity] = {}
            for snapshot in snapshots:
                for directory, identity in snapshot.ancestors:
                    previous = ancestors.setdefault(directory, identity)
                    if previous != identity:
                        raise ValueError("private control ancestor changed")
            for directory, identity in sorted(
                ancestors.items(), key=lambda item: (len(item[0].parts), str(item[0]))
            ):
                guards.append(_open_windows_private_path_guard(directory, directory=True))
                _require_staging_directory(directory)
                if _staging_identity(directory) != identity:
                    raise ValueError("private control parent changed")
            for snapshot in snapshots:
                guards.append(
                    _open_windows_private_path_guard(snapshot.source, directory=False)
                )
                _verify_fresh_control_snapshot(private_root, snapshot)
        else:
            for snapshot in snapshots:
                _verify_fresh_control_snapshot(private_root, snapshot)
        yield
    finally:
        for guard in reversed(guards):
            try:
                guard.close()
            except Exception:
                pass


class _WindowsFileDispositionInfo(ctypes.Structure):
    _fields_ = [("delete_file", ctypes.c_ubyte)]


class _WindowsFileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("creation_time", ctypes.c_longlong),
        ("last_access_time", ctypes.c_longlong),
        ("last_write_time", ctypes.c_longlong),
        ("change_time", ctypes.c_longlong),
        ("file_attributes", wintypes.DWORD),
    ]


class _WindowsStagingHandle:
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def metadata(self) -> os.stat_result:
        if self._descriptor < 0:
            raise ValueError("staging handle is closed")
        try:
            return os.fstat(self._descriptor)
        except OSError as exc:
            raise ValueError("staging handle is unavailable") from exc

    def mark_for_deletion(self) -> None:
        if self._descriptor < 0:
            raise ValueError("staging handle is closed")
        try:
            import msvcrt

            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            set_information = kernel32.SetFileInformationByHandle
            set_information.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
            )
            set_information.restype = wintypes.BOOL
            basic = _WindowsFileBasicInfo(0, 0, 0, 0, 0x00000080)
            if not set_information(
                wintypes.HANDLE(msvcrt.get_osfhandle(self._descriptor)),
                0,  # FileBasicInfo: clear FILE_ATTRIBUTE_READONLY by handle.
                ctypes.byref(basic),
                ctypes.sizeof(basic),
            ):
                raise OSError(ctypes.get_last_error(), "SetFileInformationByHandle failed")
            disposition = _WindowsFileDispositionInfo(1)
            if not set_information(
                wintypes.HANDLE(msvcrt.get_osfhandle(self._descriptor)),
                4,  # FileDispositionInfo
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise OSError(ctypes.get_last_error(), "SetFileInformationByHandle failed")
        except (AttributeError, OSError) as exc:
            raise ValueError("tracked staging entry cannot be removed safely") from exc

    def close(self) -> None:
        if self._descriptor < 0:
            return
        try:
            os.close(self._descriptor)
        finally:
            self._descriptor = -1


def _open_windows_staging_handle(
    path: Path, *, directory: bool, for_delete: bool
) -> _WindowsStagingHandle:
    """Open a Windows staging node by handle, without following its final reparse point."""

    if os.name != "nt":
        raise ValueError("handle-bound staging cleanup is unavailable")
    raw_handle: int | None = None
    try:
        import msvcrt

        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        flags = 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= 0x02000000  # FILE_FLAG_BACKUP_SEMANTICS
        access = 0x00000080  # FILE_READ_ATTRIBUTES
        if for_delete:
            access |= 0x00010100  # DELETE | FILE_WRITE_ATTRIBUTES
        handle = create_file(
            str(path),
            access,
            0x00000001,  # FILE_SHARE_READ only: deny competing write/delete opens
            None,
            3,  # OPEN_EXISTING
            flags,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        raw_handle = int(handle)
        descriptor = msvcrt.open_osfhandle(
            raw_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
        raw_handle = None  # Descriptor ownership transferred to msvcrt.
        return _WindowsStagingHandle(descriptor)
    except (AttributeError, OSError) as exc:
        raise ValueError("tracked staging entry cannot be opened safely") from exc
    finally:
        if raw_handle is not None:
            try:
                ctypes.WinDLL("Kernel32.dll", use_last_error=True).CloseHandle(
                    wintypes.HANDLE(raw_handle)
                )
            except Exception:
                pass


def _verify_handle_staging_identity(
    handle: _WindowsStagingHandle,
    expected: _StagingIdentity | _PrivateFileIdentity,
    *,
    directory: bool,
) -> None:
    metadata = handle.metadata()
    attributes = getattr(metadata, "st_file_attributes", 0)
    if bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError("tracked staging entry became a reparse point")
    if directory:
        actual: _StagingIdentity | _PrivateFileIdentity = _StagingIdentity(
            int(metadata.st_dev), int(metadata.st_ino)
        )
        valid = stat.S_ISDIR(metadata.st_mode)
    else:
        actual = _file_identity_from_metadata(metadata)
        valid = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
    if not valid or actual != expected:
        raise ValueError("tracked staging entry changed")


def _uses_posix_staging_cleanup() -> bool:
    return os.name != "nt"


def _require_posix_fd_cleanup_support() -> None:
    """Fail closed unless cleanup can keep every deletion below an open parent."""

    supported = getattr(os, "supports_dir_fd", frozenset())
    if (
        not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_NONBLOCK", 0)
        or os.open not in supported
        or os.unlink not in supported
        or os.rmdir not in supported
    ):
        raise ValueError("fd-relative staging cleanup is unavailable")


def _verify_posix_staging_descriptor_identity(
    descriptor: int,
    expected: _StagingIdentity | _PrivateFileIdentity,
    *,
    directory: bool,
) -> None:
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("tracked staging entry is unavailable") from exc
    if directory:
        actual: _StagingIdentity | _PrivateFileIdentity = _StagingIdentity(
            int(metadata.st_dev), int(metadata.st_ino)
        )
        valid = (
            stat.S_ISDIR(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o700
        )
    else:
        actual = _file_identity_from_metadata(metadata)
        valid = (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and not (stat.S_IMODE(metadata.st_mode) & 0o077)
        )
    if not valid or actual != expected:
        raise ValueError("tracked staging entry changed")


def _staging_cleanup_entry_name(path: Path, parent: Path) -> str:
    candidate = Path(path)
    name = candidate.name
    if (
        candidate.parent != Path(parent)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or ":" in name
    ):
        raise ValueError("tracked staging entry is invalid")
    return name


def _open_posix_staging_directory(
    path: Path, expected: _StagingIdentity
) -> int:
    _require_posix_fd_cleanup_support()
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("tracked staging directory cannot be opened safely") from exc
    try:
        _verify_posix_staging_descriptor_identity(
            descriptor, expected, directory=True
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_posix_staging_entry(
    parent_descriptor: int,
    name: str,
    expected: _StagingIdentity | _PrivateFileIdentity,
    *,
    directory: bool,
) -> int:
    _require_posix_fd_cleanup_support()
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if directory:
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError("tracked staging entry cannot be opened safely") from exc
    try:
        _verify_posix_staging_descriptor_identity(
            descriptor, expected, directory=directory
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _write_new_private_file(path: Path, content: bytes) -> None:
    destination = Path(path)
    if not isinstance(content, bytes):
        raise TypeError("private file content is invalid")
    _require_staging_directory(destination.parent)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise ValueError("staging file cannot be created exclusively") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        # Leave a partial staging file for handle-bound cleanup rather than
        # re-resolving a path that could have been parent-swapped.
        raise
    try:
        os.chmod(destination, 0o400)
    except OSError as exc:
        raise ValueError("staging file permissions cannot be set") from exc
    _require_staging_regular_file(destination)


def _write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_private_file(Path(path), canonical_json_bytes(value))


def _load_canonical_staged_object(path: Path, *, label: str) -> dict[str, Any]:
    _require_staging_regular_file(path)
    try:
        content = Path(path).read_bytes()
        value = strict_load_json(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if type(value) is not dict or content != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def _copy_fresh_controls(
    *,
    private_root: Path,
    price_path: Path,
    policy_path: Path,
    prompts_path: Path,
    staging: Path,
) -> tuple[Path, Path, Path]:
    price_snapshot = _capture_fresh_control(private_root, price_path)
    policy_snapshot = _capture_fresh_control(private_root, policy_path)
    prompts_snapshot = _capture_fresh_control(private_root, prompts_path)

    def copy_checked(
        snapshot: _FreshControlSnapshot, destination: Path, label: str
    ) -> Path:
        _verify_fresh_control_snapshot(private_root, snapshot)
        try:
            digest = sha256_file(snapshot.source)
        except OSError as exc:
            raise ValueError("private control file cannot be read") from exc
        _verify_fresh_control_snapshot(private_root, snapshot)
        _copy_sealed_file(
            snapshot.source,
            destination,
            {"bytes": digest.bytes, "sha256": digest.sha256},
            label=label,
        )
        _verify_fresh_control_snapshot(private_root, snapshot)
        _require_staging_regular_file(destination)
        return destination

    snapshots = (price_snapshot, policy_snapshot, prompts_snapshot)
    with _pinned_fresh_controls(private_root, snapshots):
        return (
            copy_checked(
                price_snapshot,
                staging / "control" / "price-evidence.json",
                "fresh price evidence",
            ),
            copy_checked(
                policy_snapshot,
                staging / "control" / "standing-authorization.json",
                "fresh standing authorization",
            ),
            copy_checked(
                prompts_snapshot,
                staging / "validation" / ".validation-prompts.json",
                "fresh validation prompts",
            ),
        )


def _parse_target_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or len(value) != 20 or not value.endswith("Z"):
        raise ValueError(f"{label} is invalid")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc


def _load_fresh_controls(
    *,
    price_path: Path,
    policy_path: Path,
    prompts_path: Path,
    created_at_utc: str,
    expires_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    created = _parse_target_timestamp(created_at_utc, label="created_at_utc")
    expires = _parse_target_timestamp(expires_at_utc, label="expires_at_utc")
    if expires <= created:
        raise ValueError("target expiry is invalid")
    price_value = _load_canonical_staged_object(price_path, label="fresh price evidence")
    policy_value = _load_canonical_staged_object(policy_path, label="fresh standing authorization")
    prompts_value = _load_canonical_staged_object(prompts_path, label="validation prompts")
    try:
        price = PriceEvidence.from_dict(price_value, now=created_at_utc)
        policy = StandingAuthorization.from_dict(policy_value, now=created_at_utc)
    except Exception:
        raise ValueError("fresh controls are invalid") from None
    if (
        _parse_target_timestamp(price.expires_at_utc, label="price expiry") < expires
        or _parse_target_timestamp(policy.expires_at_utc, label="policy expiry") < expires
    ):
        raise ValueError("fresh controls do not outlive the target")
    if (
        policy.training_max_usd != "6.0000"
        or policy.validation_allocation_usd != "1.2500"
        or policy.cumulative_cap_usd != "12.0000"
    ):
        raise ValueError("fresh standing authorization does not match the fixed contract")
    if type(prompts_value) is not dict or len(prompts_value) != 2:
        raise ValueError("validation prompts must contain exactly two entries")
    prompts: dict[str, str] = {}
    for group_id, prompt in prompts_value.items():
        if type(group_id) is not str or type(prompt) is not str:
            raise ValueError("validation prompts are invalid")
        prompts[group_id] = prompt
    return price.to_dict(), policy.to_dict(), prompts


def _resolved_staged_groups(
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    train_ids = set(quality_summary["accepted_train_group_ids"])
    holdout_ids = set(quality_summary["accepted_holdout_group_ids"])
    groups: list[dict[str, Any]] = []
    for group in structural_report["groups"]:
        group_id = group["group_id"]
        if group_id in train_ids:
            split = "train"
        elif group_id in holdout_ids:
            split = "holdout"
        else:
            raise ValueError("accepted group split is invalid")
        groups.append(
            {
                "group_id": group_id,
                "split": split,
                "files": [
                    {**record, "path": candidate_dir / record["name"]}
                    for record in group["files"]
                ],
            }
        )
    if len(groups) != 17:
        raise ValueError("target dataset group count is invalid")
    return groups


def _fresh_execution_config(
    *,
    source_config: dict[str, Any],
    pilot_id: str,
    target_execution_id: str,
    created_at_utc: str,
    expires_at_utc: str,
    dataset_path: Path,
    archive_path: Path,
    policy_path: Path,
    price_path: Path,
) -> dict[str, Any]:
    config = {
        "schema_version": "a2v-execution-config-v2",
        "canonical_json_version": 1,
        "execution_id": target_execution_id,
        "pilot_id": pilot_id,
        "ledger_id": source_config["ledger_id"],
        "created_at_utc": created_at_utc,
        "expires_at_utc": expires_at_utc,
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "trigger_phrase": source_config["trigger_phrase"],
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
        "negative_prompt": source_config["negative_prompt"],
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
        "dataset_manifest_sha256": sha256_file(dataset_path).sha256,
        "training_archive_sha256": sha256_file(archive_path).sha256,
        "standing_authorization_sha256": sha256_file(policy_path).sha256,
        "price_evidence_sha256": sha256_file(price_path).sha256,
        "price_source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "rate_usd_per_step": "0.006",
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
    }
    return validate_execution_config(config)


def _target_plan_bytes() -> bytes:
    return (
        "# Private A2V plan\n\n"
        "One 1,000-step A2V run.\n\n"
        "Training ceiling: $6.0000.\n"
        "Validation allocation ceiling: $1.2500.\n"
        "Cumulative ceiling: $12.0000.\n\n"
        "No third-party lip-sync is included.\n"
    ).encode("utf-8")


def _root_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "plan": sha256_file(run_dir / "plan.md"),
        "standing_authorization": sha256_file(
            run_dir / "control" / "standing-authorization.json"
        ),
        "price_evidence": sha256_file(run_dir / "control" / "price-evidence.json"),
        "structural_report": sha256_file(run_dir / "control" / "structural-report.json"),
        "quality_attestation": sha256_file(
            run_dir / "control" / "quality-attestation.json"
        ),
        "dataset_manifest": sha256_file(run_dir / "bundle" / "dataset-manifest.json"),
        "training_archive": sha256_file(run_dir / "bundle" / "training-data.zip"),
        "execution_config": sha256_file(run_dir / "control" / "execution-config.json"),
        "provider_validation_selection": sha256_file(
            run_dir / "validation" / "provider-validation-selection.json"
        ),
    }


def _walk_staging_tree(
    root: Path, *, capture_file_identities: bool
) -> tuple[list[Path], list[Path], list[_TrackedStagingFile]]:
    root = Path(root)
    _require_staging_directory(root)
    files: list[Path] = []
    directories: list[Path] = []
    file_records: list[_TrackedStagingFile] = []

    def visit(directory: Path) -> None:
        _require_staging_directory(directory)
        directories.append(directory)
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError("staging directory is unavailable") from exc
        for entry in entries:
            if (
                not entry.name
                or entry.name in {".", ".."}
                or Path(entry.name).name != entry.name
                or ":" in entry.name
            ):
                raise ValueError("staging tree entry is invalid")
            child = directory / entry.name
            if child.parent != directory:
                raise ValueError("staging tree escaped its root")
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise ValueError("staging tree entry is unavailable") from exc
            if _is_reparse_or_link(child):
                raise ValueError("staging tree aliases are prohibited")
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                file_metadata = _require_staging_regular_file(child)
                files.append(child)
                if capture_file_identities:
                    file_records.append(
                        _TrackedStagingFile(
                            path=child,
                            identity=_file_identity_from_metadata(file_metadata),
                        )
                    )
            else:
                raise ValueError("staging tree node is invalid")

    visit(root)
    return files, directories, file_records


def _safe_staging_walk(root: Path) -> tuple[list[Path], list[Path]]:
    files, directories, _ = _walk_staging_tree(root, capture_file_identities=False)
    return files, directories


def _snapshot_staging_cleanup_tree(
    root: Path,
) -> tuple[list[Path], list[Path], list[_TrackedStagingFile]]:
    return _walk_staging_tree(root, capture_file_identities=True)


def _validate_staging_layout(staging: Path) -> None:
    files, directories = _safe_staging_walk(staging)
    expected_directories = {
        staging,
        staging / "candidates",
        staging / "control",
        staging / "validation",
        staging / "bundle",
    }
    if set(directories) != expected_directories:
        raise ValueError("staging directory layout is invalid")
    if {path.name for path in staging.iterdir()} != _ROOT_LAYOUT:
        raise ValueError("staging root layout is invalid")
    if {path.name for path in (staging / "control").iterdir()} != _CONTROL_LAYOUT:
        raise ValueError("staging control layout is invalid")
    if {path.name for path in (staging / "validation").iterdir()} != _VALIDATION_LAYOUT:
        raise ValueError("staging validation layout is invalid")
    if {path.name for path in (staging / "bundle").iterdir()} != _BUNDLE_LAYOUT:
        raise ValueError("staging bundle layout is invalid")
    candidate_names = {path.name for path in (staging / "candidates").iterdir()}
    if len(candidate_names) != 68:
        raise ValueError("staging candidate count is invalid")
    expected_files = {staging / "plan.md"}
    expected_files.update((staging / "candidates" / name) for name in candidate_names)
    expected_files.update((staging / "control" / name) for name in _CONTROL_LAYOUT)
    expected_files.update((staging / "validation" / name) for name in _VALIDATION_LAYOUT)
    expected_files.update((staging / "bundle" / name) for name in _BUNDLE_LAYOUT)
    if set(files) != expected_files:
        raise ValueError("staging file layout is invalid")


def _validate_complete_staging(
    *,
    staging: Path,
    expected_bundle_id: str,
    repository_commit: str,
) -> None:
    _validate_staging_layout(staging)
    structural = _load_canonical_staged_object(
        staging / "control" / "structural-report.json", label="structural report"
    )
    attestation = _load_canonical_staged_object(
        staging / "control" / "quality-attestation.json", label="quality attestation"
    )
    dataset = _load_canonical_staged_object(
        staging / "bundle" / "dataset-manifest.json", label="dataset manifest"
    )
    config = validate_execution_config(
        _load_canonical_staged_object(
            staging / "control" / "execution-config.json", label="execution configuration"
        )
    )
    price = _load_canonical_staged_object(
        staging / "control" / "price-evidence.json", label="price evidence"
    )
    policy = _load_canonical_staged_object(
        staging / "control" / "standing-authorization.json", label="standing authorization"
    )
    fresh_structural = validate_a2v_directory(
        staging / "candidates",
        spec=A2VSpec(min_groups=17),
        trigger_phrase=config["trigger_phrase"],
    )
    if canonical_json_bytes(fresh_structural) != canonical_json_bytes(structural):
        raise ValueError("staging structural report changed")
    quality_summary = validate_quality_and_splits(attestation, fresh_structural)
    _require_dataset_contract(staging, fresh_structural, quality_summary)
    archive_path = staging / "bundle" / "training-data.zip"
    archive_digest = inspect_training_archive(archive_path, dataset.get("training_members", []))
    rebuilt_dataset = build_dataset_manifest(
        fresh_structural,
        attestation,
        archive_digest,
        candidate_dir=staging / "candidates",
    )
    if canonical_json_bytes(rebuilt_dataset) != canonical_json_bytes(dataset):
        raise ValueError("staging dataset manifest changed")
    if (
        dataset.get("counts") != {"train_groups": 12, "holdout_groups": 5}
        or type(dataset.get("training_members")) is not list
        or len(dataset["training_members"]) != 48
    ):
        raise ValueError("staging split contract is invalid")
    expected_config_digests = {
        "dataset_manifest_sha256": sha256_file(
            staging / "bundle" / "dataset-manifest.json"
        ).sha256,
        "training_archive_sha256": sha256_file(archive_path).sha256,
        "standing_authorization_sha256": sha256_file(
            staging / "control" / "standing-authorization.json"
        ).sha256,
        "price_evidence_sha256": sha256_file(
            staging / "control" / "price-evidence.json"
        ).sha256,
    }
    if any(config[field] != digest for field, digest in expected_config_digests.items()):
        raise ValueError("staging configuration binding is invalid")
    selection = _load_canonical_staged_object(
        staging / "validation" / "provider-validation-selection.json",
        label="provider validation selection",
    )
    validate_provider_validation_selection(
        selection,
        fresh_structural,
        quality_summary,
        config,
        staging / "candidates",
    )
    root = _load_canonical_staged_object(
        staging / "bundle" / "bundle-manifest.json", label="root manifest"
    )
    rebuilt_root = build_root_manifest(
        execution_id=config["execution_id"],
        created_at_utc=config["created_at_utc"],
        expires_at_utc=config["expires_at_utc"],
        repository_commit=repository_commit,
        artifacts=_root_artifacts(staging),
        holdout_groups=dataset["groups"]["holdout"],
    )
    if canonical_json_bytes(root) != canonical_json_bytes(rebuilt_root):
        raise ValueError("staging root manifest changed")
    if compute_bundle_id(root) != expected_bundle_id:
        raise ValueError("staging bundle identity changed")
    try:
        fresh_price = PriceEvidence.from_dict(price, now=config["created_at_utc"])
        fresh_policy = StandingAuthorization.from_dict(
            policy, now=config["created_at_utc"]
        )
        target_expiry = _parse_target_timestamp(
            config["expires_at_utc"], label="target expiry"
        )
        if (
            _parse_target_timestamp(fresh_price.expires_at_utc, label="price expiry")
            < target_expiry
            or _parse_target_timestamp(fresh_policy.expires_at_utc, label="policy expiry")
            < target_expiry
        ):
            raise ValueError("staging controls do not outlive the target")
        if (
            fresh_policy.training_max_usd != "6.0000"
            or fresh_policy.validation_allocation_usd != "1.2500"
            or fresh_policy.cumulative_cap_usd != "12.0000"
        ):
            raise ValueError("staging policy does not match the fixed contract")
    except ValueError:
        raise
    except Exception:
        raise ValueError("staging controls are invalid") from None


def _tracked_directory_at(
    staging: _TrackedStaging, original: Path, current_root: Path
) -> Path:
    try:
        relative = Path(original).relative_to(staging.path)
    except ValueError as exc:
        raise ValueError("tracked staging directory escaped its root") from exc
    if not relative.parts:
        return Path(current_root)
    return Path(current_root).joinpath(*relative.parts)


def _verify_tracked_staging_directories(
    staging: _TrackedStaging, *, current_root: Path
) -> None:
    if staging.directories.get(staging.path) != staging.identity:
        raise ValueError("tracked staging root is invalid")
    for original, expected in staging.directories.items():
        current = _tracked_directory_at(staging, original, current_root)
        _require_staging_directory(current)
        if _staging_identity(current) != expected:
            raise ValueError("tracked staging directory changed")


def _remove_windows_tracked_staging_file(
    staging: _TrackedStaging, record: _TrackedStagingFile
) -> None:
    parent = record.path.parent
    expected_parent = staging.directories.get(parent)
    if expected_parent is None:
        raise ValueError("staging file parent is untracked")
    parent_handle = _open_windows_staging_handle(
        parent, directory=True, for_delete=False
    )
    file_handle: _WindowsStagingHandle | None = None
    try:
        _verify_handle_staging_identity(
            parent_handle, expected_parent, directory=True
        )
        file_handle = _open_windows_staging_handle(
            record.path, directory=False, for_delete=True
        )
        _verify_handle_staging_identity(
            file_handle, record.identity, directory=False
        )
        file_handle.mark_for_deletion()
    finally:
        if file_handle is not None:
            file_handle.close()
        parent_handle.close()


def _remove_windows_tracked_staging_directory(
    staging: _TrackedStaging, directory: Path
) -> None:
    target = Path(directory)
    expected_target = staging.directories.get(target)
    if expected_target is None:
        raise ValueError("staging directory is untracked")
    if target == staging.path:
        parent = staging.parent
        expected_parent = staging.parent_identity
    else:
        parent = target.parent
        expected_parent = staging.directories.get(parent)
        if expected_parent is None:
            raise ValueError("staging directory parent is untracked")
    parent_handle = _open_windows_staging_handle(
        parent, directory=True, for_delete=False
    )
    target_handle: _WindowsStagingHandle | None = None
    try:
        _verify_handle_staging_identity(
            parent_handle, expected_parent, directory=True
        )
        target_handle = _open_windows_staging_handle(
            target, directory=True, for_delete=True
        )
        _verify_handle_staging_identity(
            target_handle, expected_target, directory=True
        )
        target_handle.mark_for_deletion()
    finally:
        if target_handle is not None:
            target_handle.close()
        parent_handle.close()


def _remove_posix_tracked_staging_file(
    staging: _TrackedStaging, record: _TrackedStagingFile
) -> None:
    parent = record.path.parent
    expected_parent = staging.directories.get(parent)
    if expected_parent is None:
        raise ValueError("staging file parent is untracked")
    name = _staging_cleanup_entry_name(record.path, parent)
    parent_descriptor = _open_posix_staging_directory(parent, expected_parent)
    file_descriptor: int | None = None
    try:
        file_descriptor = _open_posix_staging_entry(
            parent_descriptor, name, record.identity, directory=False
        )
        os.unlink(name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError("tracked staging entry cannot be removed safely") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(parent_descriptor)


def _remove_posix_tracked_staging_directory(
    staging: _TrackedStaging, directory: Path
) -> None:
    target = Path(directory)
    expected_target = staging.directories.get(target)
    if expected_target is None:
        raise ValueError("staging directory is untracked")
    if target == staging.path:
        parent = staging.parent
        expected_parent = staging.parent_identity
    else:
        parent = target.parent
        expected_parent = staging.directories.get(parent)
        if expected_parent is None:
            raise ValueError("staging directory parent is untracked")
    name = _staging_cleanup_entry_name(target, parent)
    parent_descriptor = _open_posix_staging_directory(parent, expected_parent)
    target_descriptor: int | None = None
    try:
        target_descriptor = _open_posix_staging_entry(
            parent_descriptor, name, expected_target, directory=True
        )
        os.rmdir(name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError("tracked staging entry cannot be removed safely") from exc
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(parent_descriptor)


def _remove_tracked_staging_file(
    staging: _TrackedStaging, record: _TrackedStagingFile
) -> None:
    if os.name == "nt":
        _remove_windows_tracked_staging_file(staging, record)
    else:
        _remove_posix_tracked_staging_file(staging, record)


def _remove_tracked_staging_directory(
    staging: _TrackedStaging, directory: Path
) -> None:
    if os.name == "nt":
        _remove_windows_tracked_staging_directory(staging, directory)
    else:
        _remove_posix_tracked_staging_directory(staging, directory)


def _remove_staged_prompt_file(path: Path, staging: _TrackedStaging) -> None:
    candidate = Path(path)
    if candidate.parent != staging.path / "validation":
        raise ValueError("staged prompt file is invalid")
    metadata = _require_staging_regular_file(candidate)
    _remove_tracked_staging_file(
        staging,
        _TrackedStagingFile(
            path=candidate,
            identity=_file_identity_from_metadata(metadata),
        ),
    )


def _create_tracked_staging(target: Path) -> _TrackedStaging:
    destination = Path(target)
    parent = destination.parent
    if _uses_posix_staging_cleanup():
        _require_posix_fd_cleanup_support()
    _require_staging_directory(parent)
    if _has_alias_component(parent) or _has_case_alias(parent):
        raise ValueError("canonical runs directory is required")
    try:
        path = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=parent))
        if os.name != "nt":
            os.chmod(path, 0o700)
        _require_staging_directory(path)
    except Exception:
        raise ValueError("private staging directory cannot be created") from None
    parent_identity = _staging_identity(parent)
    identity = _staging_identity(path)
    return _TrackedStaging(
        path=path,
        parent=parent,
        parent_identity=parent_identity,
        identity=identity,
        directories={path: identity},
    )


def _cleanup_tracked_staging(staging: _TrackedStaging) -> None:
    path = staging.path
    if (
        path.parent != staging.parent
        or _has_alias_component(staging.parent)
        or _has_case_alias(staging.parent)
    ):
        raise ValueError("tracked staging directory changed")
    _verify_tracked_staging_directories(staging, current_root=path)
    _files, directories, file_records = _snapshot_staging_cleanup_tree(path)
    if set(directories) != set(staging.directories):
        raise ValueError("tracked staging directory layout changed")
    try:
        for record in file_records:
            _remove_tracked_staging_file(staging, record)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _remove_tracked_staging_directory(staging, directory)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("tracked staging directory cannot be removed safely") from exc


@contextmanager
def _fresh_private_staging(target: Path) -> Iterator[_TrackedStaging]:
    staging = _create_tracked_staging(target)
    try:
        yield staging
    except BaseException:
        if not staging.published:
            _cleanup_tracked_staging(staging)
        raise
    else:
        if not staging.published:
            _cleanup_tracked_staging(staging)


def _target_absent(parent: Path, target_name: str) -> bool:
    try:
        with os.scandir(parent) as entries:
            return not any(entry.name.casefold() == target_name.casefold() for entry in entries)
    except OSError:
        return False


def _move_directory_no_replace(staging: Path, target: Path) -> bool:
    """Use the Windows same-volume no-replace move primitive, or fail closed."""

    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
        move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move_file_ex.restype = ctypes.c_int
        return bool(move_file_ex(str(staging), str(target), _MOVEFILE_WRITE_THROUGH))
    except (AttributeError, OSError):
        return False


def _publish_new_run_no_replace(
    staging: Path,
    target: Path,
    *,
    tracked: _TrackedStaging,
    expected_bundle_id: str,
    repository_commit: str,
) -> None:
    """Publish one already-validated staging tree without replacement semantics."""

    staged = Path(staging)
    destination = Path(target)
    if not isinstance(tracked, _TrackedStaging):
        raise ValueError("publication validation inputs are invalid")
    if (
        staged.parent != destination.parent
        or not staged.name.startswith(_STAGING_PREFIX)
        or staged == destination
    ):
        raise ValueError("publication paths are invalid")
    _require_staging_directory(staged.parent)
    _require_staging_directory(staged)
    staged_identity = _staging_identity(staged)
    if tracked.path != staged or tracked.parent != staged.parent:
        raise ValueError("tracked staging path changed")
    _verify_tracked_staging_directories(tracked, current_root=staged)
    if staged_identity != tracked.identity:
        raise ValueError("tracked staging directory changed")
    _validate_complete_staging(
        staging=staged,
        expected_bundle_id=expected_bundle_id,
        repository_commit=repository_commit,
    )
    _verify_tracked_staging_directories(tracked, current_root=staged)
    if (
        _has_alias_component(staged.parent)
        or _has_case_alias(staged.parent)
        or not _target_absent(
            staged.parent, destination.name
        )
    ):
        raise ValueError("new target directory is unavailable")
    if not _move_directory_no_replace(staged, destination):
        raise ValueError("new target directory cannot be published")
    # The context manager must never clean an ambiguous post-move target.
    tracked.published = True
    try:
        _require_staging_directory(destination)
        if _staging_identity(destination) != staged_identity:
            raise ValueError("new target directory changed after publication")
        _verify_tracked_staging_directories(tracked, current_root=destination)
        _validate_complete_staging(
            staging=destination,
            expected_bundle_id=expected_bundle_id,
            repository_commit=repository_commit,
        )
        if _staging_identity(destination) != staged_identity:
            raise ValueError("new target directory changed after publication")
        _verify_tracked_staging_directories(tracked, current_root=destination)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("new target directory cannot be published") from exc


def refresh_sealed_a2v_run(
    *,
    private_root: Path,
    pilot_id: str,
    source_execution_id: str,
    expected_source_bundle_id: str,
    target_execution_id: str,
    created_at_utc: str,
    expires_at_utc: str,
    fresh_price_evidence_path: Path,
    fresh_standing_authorization_path: Path,
    validation_prompts_path: Path,
    repository_commit: str,
) -> FreshA2VRunResult:
    """Build and publish one fresh, immutable A2V target without paid execution."""

    snapshot = verify_source_run_static(
        private_root=private_root,
        pilot_id=pilot_id,
        source_execution_id=source_execution_id,
        expected_source_bundle_id=expected_source_bundle_id,
    )
    if source_execution_id == target_execution_id:
        raise ValueError("source and target execution IDs must differ")
    target = canonical_new_run_dir(private_root, pilot_id, target_execution_id)
    with _fresh_private_staging(target) as tracked:
        staging = tracked.path
        for directory in ("candidates", "control", "validation", "bundle"):
            path = staging / directory
            try:
                path.mkdir(mode=0o700)
                if os.name != "nt":
                    os.chmod(path, 0o700)
            except OSError as exc:
                raise ValueError("private staging layout cannot be created") from exc
            _require_staging_directory(path)
            tracked.directories[path] = _staging_identity(path)

        staged_price, staged_policy, staged_prompts = _copy_fresh_controls(
            private_root=Path(private_root),
            price_path=fresh_price_evidence_path,
            policy_path=fresh_standing_authorization_path,
            prompts_path=validation_prompts_path,
            staging=staging,
        )
        _price, _policy, prompts = _load_fresh_controls(
            price_path=staged_price,
            policy_path=staged_policy,
            prompts_path=staged_prompts,
            created_at_utc=created_at_utc,
            expires_at_utc=expires_at_utc,
        )
        structural, attestation = copy_accepted_candidates(snapshot, staging / "candidates")
        quality_summary = validate_quality_and_splits(attestation, structural)
        _require_dataset_contract(staging, structural, quality_summary)
        _write_new_canonical_json(staging / "control" / "structural-report.json", structural)
        _write_new_canonical_json(staging / "control" / "quality-attestation.json", attestation)

        groups = _resolved_staged_groups(structural, quality_summary, staging / "candidates")
        archive_path = staging / "bundle" / "training-data.zip"
        archive_digest = build_training_archive(groups, archive_path)
        dataset = build_dataset_manifest(
            structural,
            attestation,
            archive_digest,
            candidate_dir=staging / "candidates",
        )
        _write_new_canonical_json(staging / "bundle" / "dataset-manifest.json", dataset)
        config = _fresh_execution_config(
            source_config=snapshot.source_config,
            pilot_id=pilot_id,
            target_execution_id=target_execution_id,
            created_at_utc=created_at_utc,
            expires_at_utc=expires_at_utc,
            dataset_path=staging / "bundle" / "dataset-manifest.json",
            archive_path=archive_path,
            policy_path=staged_policy,
            price_path=staged_price,
        )
        _write_new_canonical_json(staging / "control" / "execution-config.json", config)
        selection = build_provider_validation_selection(
            structural_report=structural,
            quality_summary=quality_summary,
            execution_config=config,
            candidate_dir=staging / "candidates",
            prompts=prompts,
        )
        _write_new_canonical_json(
            staging / "validation" / "provider-validation-selection.json", selection
        )
        validate_provider_validation_selection(
            _load_canonical_staged_object(
                staging / "validation" / "provider-validation-selection.json",
                label="provider validation selection",
            ),
            structural,
            quality_summary,
            _load_canonical_staged_object(
                staging / "control" / "execution-config.json",
                label="execution configuration",
            ),
            staging / "candidates",
        )
        _remove_staged_prompt_file(staged_prompts, tracked)
        _write_new_private_file(staging / "plan.md", _target_plan_bytes())
        root = build_root_manifest(
            execution_id=target_execution_id,
            created_at_utc=created_at_utc,
            expires_at_utc=expires_at_utc,
            repository_commit=repository_commit,
            artifacts=_root_artifacts(staging),
            holdout_groups=dataset["groups"]["holdout"],
        )
        _write_new_canonical_json(staging / "bundle" / "bundle-manifest.json", root)
        bundle_id = compute_bundle_id(root)
        _validate_complete_staging(
            staging=staging,
            expected_bundle_id=bundle_id,
            repository_commit=repository_commit,
        )
        _publish_new_run_no_replace(
            staging,
            target,
            tracked=tracked,
            expected_bundle_id=bundle_id,
            repository_commit=repository_commit,
        )
        return FreshA2VRunResult(
            execution_id=target_execution_id,
            bundle_id=bundle_id,
            run_dir=target,
        )
