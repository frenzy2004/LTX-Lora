import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


@dataclass(frozen=True)
class FileDigest:
    name: str
    bytes: int
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    _reject_unsupported(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _reject_unsupported(value: Any) -> None:
    if isinstance(value, float):
        raise TypeError("floats are prohibited in canonical JSON")
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON dictionary keys must be strings")
            _reject_unsupported(item)
    elif type(value) is list:
        for item in value:
            _reject_unsupported(item)
    elif value is not None and type(value) not in (bool, int, str):
        raise TypeError(
            f"unsupported canonical JSON type: {type(value).__name__}"
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def strict_load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=_reject_constant,
    )


def safe_relative_name(name: str) -> str:
    if "\\" in name:
        raise ValueError("relative artifact names cannot contain backslashes")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError("relative artifact names cannot contain control characters")
    if not name.isascii():
        raise ValueError("relative artifact names must contain ASCII characters only")
    windows_path = PureWindowsPath(name)
    if windows_path.drive:
        raise ValueError("relative artifact names cannot contain a Windows drive")
    if PurePosixPath(name).is_absolute() or windows_path.is_absolute():
        raise ValueError("artifact name must be relative")
    if ".." in name.split("/"):
        raise ValueError("relative artifact names cannot contain '..'")
    return name


def sha256_file(path: Path) -> FileDigest:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return FileDigest(
        name=path.name,
        bytes=byte_count,
        sha256=digest.hexdigest(),
    )


def atomic_write_json(path: Path, value: Any) -> None:
    content = canonical_json_bytes(value)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
