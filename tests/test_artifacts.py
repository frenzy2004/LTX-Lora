import hashlib
from pathlib import Path

import pytest

from ltx_lora_pilot.artifacts import (
    FileDigest,
    atomic_write_json,
    canonical_json_bytes,
    safe_relative_name,
    sha256_file,
    strict_load_json,
)


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": "0.0002"}) == b'{"a":"0.0002","b":2}'


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"cap":"12.0000","cap":"14.0000"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_load_json(path)


def test_canonical_json_rejects_float() -> None:
    with pytest.raises(TypeError, match="floats are prohibited"):
        canonical_json_bytes({"learning_rate": 0.0002})


def test_canonical_json_rejects_non_string_dictionary_keys() -> None:
    with pytest.raises(TypeError, match="dictionary keys must be strings"):
        canonical_json_bytes({1: "value"})


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_non_finite_numbers(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "non-finite.json"
    path.write_text(f'{{"value":{constant}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        strict_load_json(path)


def test_canonical_json_rejects_unsupported_object_types() -> None:
    with pytest.raises(TypeError, match="unsupported canonical JSON type"):
        canonical_json_bytes({"frames": (1, 2)})


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "bad\nname"])
def test_safe_relative_name_rejects_unsafe_input(name: str) -> None:
    with pytest.raises(ValueError):
        safe_relative_name(name)


def test_safe_relative_name_rejects_non_ascii_input() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        safe_relative_name("café.json")


def test_sha256_file_reports_name_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    content = (b"a" * (1024 * 1024)) + b"chunk-boundary"
    path.write_bytes(content)

    assert sha256_file(path) == FileDigest(
        name="artifact.bin",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_atomic_write_json_replaces_with_canonical_output(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("stale", encoding="utf-8")

    atomic_write_json(path, {"z": 2, "a": "0.0002"})

    assert path.read_bytes() == b'{"a":"0.0002","z":2}'
    assert list(tmp_path.iterdir()) == [path]
