from __future__ import annotations

import os
from pathlib import Path
import re
import stat


PRIVATE_ROOT_ENVIRONMENT_VARIABLE = "LTX_LORA_PRIVATE_ROOT"
UUID4_HEX = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
PILOT_ID_PATTERN = re.compile(rf"pilot_{UUID4_HEX}", re.ASCII)
EXECUTION_ID_PATTERN = re.compile(rf"exec_{UUID4_HEX}", re.ASCII)


def _is_alias_component(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except OSError:
        return True


def _has_alias_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and _is_alias_component(current):
            return True
    return False


def _has_case_alias(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        try:
            with os.scandir(current) as entries:
                matching_names = {
                    entry.name
                    for entry in entries
                    if entry.name.casefold() == part.casefold()
                }
        except OSError:
            return True
        if part not in matching_names:
            return True
        current /= part
    return False


def _has_ads_syntax(path: Path) -> bool:
    parts = path.parts[1:] if path.anchor else path.parts
    return any(":" in part for part in parts)


def _typed_id(value: str, pattern: re.Pattern[str], *, label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _canonical_absolute_path(value: Path, *, directory: bool) -> Path:
    try:
        candidate = Path(value)
    except (OSError, TypeError, ValueError):
        raise ValueError("private workspace path is invalid") from None
    raw = str(candidate)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or _has_ads_syntax(candidate)
    ):
        raise ValueError("private workspace path is invalid")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise ValueError("private workspace path is invalid")
    if _has_alias_component(candidate) or _has_case_alias(candidate):
        raise ValueError("private workspace path is invalid")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, ValueError):
        raise ValueError("private workspace path is invalid") from None
    if directory:
        valid_type = resolved.is_dir()
    else:
        valid_type = resolved.is_file()
    if not valid_type or str(resolved) != str(absolute):
        raise ValueError("private workspace path is invalid")
    return resolved


def canonical_new_run_dir(
    private_root: Path,
    pilot_id: str,
    execution_id: str,
) -> Path:
    """Derive the one absent canonical run directory for a typed execution ID."""

    root = _canonical_absolute_path(private_root, directory=True)
    pilot = _typed_id(pilot_id, PILOT_ID_PATTERN, label="pilot_id")
    execution = _typed_id(
        execution_id,
        EXECUTION_ID_PATTERN,
        label="execution_id",
    )
    runs_parent = root / "pilots" / pilot / "runs"
    canonical_parent = _canonical_absolute_path(runs_parent, directory=True)
    if str(canonical_parent) != str(runs_parent):
        raise ValueError("canonical runs directory is required")
    try:
        with os.scandir(canonical_parent) as entries:
            if any(entry.name.casefold() == execution.casefold() for entry in entries):
                raise ValueError("new run directory must not already exist")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("canonical runs directory is required") from exc
    target = canonical_parent / execution
    if target.exists() or target.is_symlink():
        raise ValueError("new run directory must not already exist")
    return target


def require_canonical_private_file(private_root: Path, path: Path) -> Path:
    """Return an existing unaliased, single-link file below the approved root."""

    root = _canonical_absolute_path(private_root, directory=True)
    candidate = _canonical_absolute_path(path, directory=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("private file must be beneath the approved root") from None
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ValueError("private file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or _is_alias_component(candidate)
        or _has_case_alias(candidate)
    ):
        raise ValueError("private file must be a canonical single-link regular file")
    return candidate


def approved_private_root_from_environment() -> Path:
    raw = os.environ.get(PRIVATE_ROOT_ENVIRONMENT_VARIABLE)
    if raw is None:
        raise ValueError("approved private root is required")
    if not raw or raw != raw.strip() or "\x00" in raw:
        raise ValueError("approved private root is invalid")
    try:
        return _canonical_absolute_path(Path(raw), directory=True)
    except ValueError:
        raise ValueError("approved private root is invalid") from None


def resolve_pilot_ledger(private_root: Path, pilot_id: str) -> Path:
    root = _canonical_absolute_path(private_root, directory=True)
    pilot = _typed_id(pilot_id, PILOT_ID_PATTERN, label="pilot_id")
    ledger = root / "pilots" / pilot / "ledger" / "pilot.sqlite3"
    return _canonical_absolute_path(ledger, directory=False)


def require_canonical_run_dir(
    private_root: Path,
    pilot_id: str,
    execution_id: str,
    run_dir: Path,
) -> Path:
    root = _canonical_absolute_path(private_root, directory=True)
    pilot = _typed_id(pilot_id, PILOT_ID_PATTERN, label="pilot_id")
    execution = _typed_id(
        execution_id,
        EXECUTION_ID_PATTERN,
        label="execution_id",
    )
    supplied = _canonical_absolute_path(run_dir, directory=True)
    expected = root / "pilots" / pilot / "runs" / execution
    if str(supplied) != str(expected):
        raise ValueError("canonical run directory is required")
    return supplied
