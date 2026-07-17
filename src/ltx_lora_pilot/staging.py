from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from types import MappingProxyType
from typing import Any, Iterator, Mapping
import uuid

from .a2v_bundle import compute_bundle_id
from .artifacts import FileDigest, canonical_json_bytes, strict_load_json
from .a2v_contracts import validate_execution_config
from .private_workspace import require_canonical_run_dir


class StagedArtifactChanged(RuntimeError):
    """Raised when a private staged input no longer matches its sealed digest."""


def _remove_stage_session(path: Path) -> None:
    def clear_readonly_and_retry(function: Any, failed_path: str, _exc_info: Any) -> None:
        os.chmod(failed_path, stat.S_IREAD | stat.S_IWRITE)
        function(failed_path)

    try:
        shutil.rmtree(path, onerror=clear_readonly_and_retry)
    except FileNotFoundError:
        return


@dataclass(frozen=True)
class StagedValidationPair:
    group_id: str
    prompt: str
    image: Path = field(repr=False)
    audio: Path = field(repr=False)


@dataclass(frozen=True)
class _StagedFile:
    path: Path = field(repr=False)
    digest: FileDigest
    identity: tuple[int, int]
    mode: int
    modified_ns: int


@dataclass
class StagedArtifactGuard:
    bundle_id: str
    execution_id: str
    training_zip: Path = field(repr=False)
    validation_pairs: tuple[StagedValidationPair, StagedValidationPair]
    execution_config: Mapping[str, Any] = field(repr=False)
    provider_selection: Mapping[str, Any] = field(repr=False)
    _stage_session: Path = field(repr=False)
    _files: tuple[_StagedFile, ...] = field(repr=False)
    _platform_guards: list[Any] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def verify_unchanged(self) -> bool:
        if self._closed:
            return False
        return all(_verify_staged_file(record) for record in self._files)

    def require_unchanged(self) -> None:
        if not self.verify_unchanged():
            raise StagedArtifactChanged("staged artifact changed before provider submission")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for guard in reversed(self._platform_guards):
            try:
                guard.close()
            except Exception:
                pass
        self._platform_guards.clear()
        _remove_stage_session(self._stage_session)


class _PosixReadGuard:
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def close(self) -> None:
        if self._descriptor < 0:
            return
        try:
            import fcntl

            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._descriptor = -1


class _WindowsReadGuard:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def close(self) -> None:
        if not self._handle:
            return
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = 0


def _open_platform_read_guard(path: Path) -> Any:
    """Hold a cooperative read lock while a staged path can be uploaded."""

    if os.name == "nt":
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
        handle = create_file(
            str(path),
            0x80000000,  # GENERIC_READ
            0x00000001,  # FILE_SHARE_READ only: deny write and delete sharing
            None,
            3,  # OPEN_EXISTING
            0x00000080,  # FILE_ATTRIBUTE_NORMAL
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise ValueError("unable to retain staged file guard")
        return _WindowsReadGuard(int(handle))

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except Exception:
        os.close(descriptor)
        raise
    return _PosixReadGuard(descriptor)


def _sha256_stream(source: Any, destination: Any | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
        if destination is not None:
            destination.write(chunk)
    return byte_count, digest.hexdigest()


def _is_safe_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and not path.is_symlink()
        and not bool(attributes & reparse)
    )


def _assert_expected_digest(
    digest: FileDigest,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if (
        type(expected) is not dict
        or type(expected.get("bytes")) is not int
        or type(expected.get("sha256")) is not str
        or digest.bytes != expected["bytes"]
        or digest.sha256 != expected["sha256"]
    ):
        raise ValueError(f"{label} does not match the sealed manifest")


def _copy_sealed_file(
    source_path: Path,
    destination_path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> _StagedFile:
    if not _is_safe_regular_file(source_path):
        raise ValueError(f"{label} source is not a safe regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} source is not a safe regular file")
        try:
            output_descriptor = os.open(
                destination_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except OSError as exc:
            raise ValueError("staging destination cannot be created exclusively") from exc
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as source, os.fdopen(
                output_descriptor,
                "wb",
                closefd=False,
            ) as destination:
                byte_count, sha256 = _sha256_stream(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            os.close(output_descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        on_path = source_path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed while staging") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (on_path.st_dev, on_path.st_ino, on_path.st_size, on_path.st_mtime_ns)
    ):
        raise ValueError(f"{label} changed while staging")
    digest = FileDigest(destination_path.name, byte_count, sha256)
    _assert_expected_digest(digest, expected, label=label)
    os.chmod(destination_path, 0o400)
    metadata = destination_path.stat()
    return _StagedFile(
        path=destination_path,
        digest=digest,
        identity=(int(metadata.st_dev), int(metadata.st_ino)),
        mode=stat.S_IMODE(metadata.st_mode),
        modified_ns=int(metadata.st_mtime_ns),
    )


def _verify_staged_file(record: _StagedFile) -> bool:
    if not _is_safe_regular_file(record.path):
        return False
    try:
        metadata = record.path.stat()
        if (
            (int(metadata.st_dev), int(metadata.st_ino)) != record.identity
            or stat.S_IMODE(metadata.st_mode) != record.mode
            or int(metadata.st_mtime_ns) != record.modified_ns
        ):
            return False
        with record.path.open("rb") as source:
            bytes_read, sha256 = _sha256_stream(source)
    except OSError:
        return False
    return bytes_read == record.digest.bytes and sha256 == record.digest.sha256


def _json_digest(path: Path) -> FileDigest:
    data = path.read_bytes()
    return FileDigest(path.name, len(data), hashlib.sha256(data).hexdigest())


def _expected_root_artifact(root: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    artifacts = root.get("artifacts")
    if type(artifacts) is not dict or type(artifacts.get(role)) is not dict:
        raise ValueError("sealed root manifest is invalid")
    return artifacts[role]


def _load_staged_json(path: Path, *, label: str) -> dict[str, Any]:
    value = strict_load_json(path)
    if type(value) is not dict or canonical_json_bytes(value) != path.read_bytes():
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _create_stage_session(private_root: Path, bundle_id: str) -> tuple[Path, Path]:
    # The bundle identity is already sealed in the manifest, request and ledger.
    # Do not repeat its 64-character digest in every staged path: doing so can
    # exceed the legacy Windows path limit before a provider request is issued.
    if not isinstance(bundle_id, str) or len(bundle_id) != 64:
        raise ValueError("confirmed bundle ID is invalid")
    parent = private_root / ".s"
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    session = parent / uuid.uuid4().hex
    os.mkdir(session, 0o700)
    return session, session


@contextmanager
def stage_bundle(
    run_dir: Path,
    *,
    approved_private_root: Path,
    confirmed_bundle_id: str,
    pilot_id: str,
    execution_id: str,
) -> Iterator[StagedArtifactGuard]:
    """Copy the exact sealed paid inputs into a unique guarded private staging area."""

    canonical_run_dir = require_canonical_run_dir(
        approved_private_root,
        pilot_id,
        execution_id,
        Path(run_dir),
    )
    if not isinstance(confirmed_bundle_id, str) or len(confirmed_bundle_id) != 64:
        raise ValueError("confirmed bundle ID is invalid")
    session, bundle_dir = _create_stage_session(
        Path(approved_private_root), confirmed_bundle_id
    )
    guard: StagedArtifactGuard | None = None
    try:
        root_record = _copy_sealed_file(
            canonical_run_dir / "bundle" / "bundle-manifest.json",
            bundle_dir / "bundle-manifest.json",
            {"bytes": (canonical_run_dir / "bundle" / "bundle-manifest.json").stat().st_size,
             "sha256": hashlib.sha256((canonical_run_dir / "bundle" / "bundle-manifest.json").read_bytes()).hexdigest()},
            label="bundle manifest",
        )
        root = _load_staged_json(root_record.path, label="bundle manifest")
        if compute_bundle_id(root) != confirmed_bundle_id:
            raise ValueError("sealed bundle ID changed before staging")

        config_record = _copy_sealed_file(
            canonical_run_dir / "control" / "execution-config.json",
            bundle_dir / "execution-config.json",
            _expected_root_artifact(root, "execution_config"),
            label="execution configuration",
        )
        config = validate_execution_config(
            _load_staged_json(config_record.path, label="execution configuration")
        )
        if config["execution_id"] != execution_id:
            raise ValueError("sealed execution ID changed before staging")

        selection_record = _copy_sealed_file(
            canonical_run_dir / "validation" / "provider-validation-selection.json",
            bundle_dir / "provider-validation-selection.json",
            _expected_root_artifact(root, "provider_validation_selection"),
            label="provider validation selection",
        )
        selection = _load_staged_json(selection_record.path, label="provider validation selection")
        config_digest = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
        if selection.get("execution_config_sha256") != config_digest:
            raise ValueError("staged validation selection is not bound to execution config")
        items = selection.get("items")
        if type(items) is not list or len(items) != 2:
            raise ValueError("staged validation selection must contain exactly two items")

        archive_record = _copy_sealed_file(
            canonical_run_dir / "bundle" / "training-data.zip",
            bundle_dir / "training-data.zip",
            _expected_root_artifact(root, "training_archive"),
            label="training archive",
        )
        if archive_record.digest.sha256 != config["training_archive_sha256"]:
            raise ValueError("staged training archive is not bound to execution config")

        staged_records = [root_record, config_record, selection_record, archive_record]
        pairs: list[StagedValidationPair] = []
        used_names: set[str] = set()
        for index, item in enumerate(items):
            if type(item) is not dict:
                raise ValueError("staged validation selection item is invalid")
            group_id = item.get("group_id")
            prompt = item.get("prompt")
            image = item.get("image")
            audio = item.get("audio")
            if not isinstance(group_id, str) or not isinstance(prompt, str):
                raise ValueError("staged validation selection item is invalid")
            if type(image) is not dict or type(audio) is not dict:
                raise ValueError("staged validation selection item is invalid")
            image_name = image.get("name")
            audio_name = audio.get("name")
            if (
                not isinstance(image_name, str)
                or not isinstance(audio_name, str)
                or image_name in used_names
                or audio_name in used_names
            ):
                raise ValueError("staged validation media selection is invalid")
            used_names.update((image_name, audio_name))
            staged_image = _copy_sealed_file(
                canonical_run_dir / "candidates" / image_name,
                bundle_dir / image_name,
                image,
                label=f"validation image {index}",
            )
            staged_audio = _copy_sealed_file(
                canonical_run_dir / "candidates" / audio_name,
                bundle_dir / audio_name,
                audio,
                label=f"validation audio {index}",
            )
            staged_records.extend((staged_image, staged_audio))
            pairs.append(
                StagedValidationPair(
                    group_id=group_id,
                    prompt=prompt,
                    image=staged_image.path,
                    audio=staged_audio.path,
                )
            )
        if len(pairs) != 2:
            raise ValueError("staged validation selection must contain exactly two pairs")
        platform_guards: list[Any] = []
        try:
            for record in staged_records:
                platform_guards.append(_open_platform_read_guard(record.path))
        except Exception:
            for acquired in reversed(platform_guards):
                try:
                    acquired.close()
                except Exception:
                    pass
            raise
        guard = StagedArtifactGuard(
            bundle_id=confirmed_bundle_id,
            execution_id=execution_id,
            training_zip=archive_record.path,
            validation_pairs=(pairs[0], pairs[1]),
            execution_config=_freeze_json(config),
            provider_selection=_freeze_json(selection),
            _stage_session=session,
            _files=tuple(staged_records),
            _platform_guards=platform_guards,
        )
        guard.require_unchanged()
        yield guard
    finally:
        if guard is not None:
            guard.close()
        else:
            _remove_stage_session(session)
