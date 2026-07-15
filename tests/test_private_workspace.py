from __future__ import annotations

from pathlib import Path

import pytest

import ltx_lora_pilot.private_workspace as private_workspace
from ltx_lora_pilot.private_workspace import (
    approved_private_root_from_environment,
    require_canonical_run_dir,
    resolve_pilot_ledger,
)


PILOT_ID = "pilot_00000000000040008000000000000001"
EXECUTION_ID = "exec_00000000000040008000000000000002"


def test_approved_private_root_is_absolute_and_independent_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    private_root.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("LTX_LORA_PRIVATE_ROOT", str(private_root))
    monkeypatch.chdir(unrelated)

    assert approved_private_root_from_environment() == private_root.resolve(strict=True)


@pytest.mark.parametrize(
    "raw",
    [None, "", "relative-root", " relative-root ", "\x00private-root"],
)
def test_approved_private_root_rejects_missing_or_unsafe_values_neutrally(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
) -> None:
    environment = {} if raw is None else {"LTX_LORA_PRIVATE_ROOT": raw}
    monkeypatch.setattr(private_workspace.os, "environ", environment)

    with pytest.raises(ValueError, match="approved private root") as exc_info:
        approved_private_root_from_environment()

    assert exc_info.value.__cause__ is None
    if raw:
        assert raw not in str(exc_info.value)


def test_approved_private_root_rejects_whitespace_missing_and_file_paths_neutrally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_file = tmp_path / "not-a-directory"
    existing_file.write_text("private", encoding="utf-8")
    candidates = [
        f" {tmp_path}",
        f"{tmp_path} ",
        str(tmp_path / "missing"),
        str(existing_file),
    ]

    for raw in candidates:
        monkeypatch.setenv("LTX_LORA_PRIVATE_ROOT", raw)
        with pytest.raises(ValueError, match="approved private root") as exc_info:
            approved_private_root_from_environment()
        assert exc_info.value.__cause__ is None
        assert raw not in str(exc_info.value)


def test_approved_private_root_rejects_alias_component_neutrally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    private_root.mkdir()
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == private_root or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setenv("LTX_LORA_PRIVATE_ROOT", str(private_root))

    with pytest.raises(ValueError, match="approved private root") as exc_info:
        approved_private_root_from_environment()

    assert str(private_root) not in str(exc_info.value)


def test_approved_private_root_rejects_case_alias_neutrally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "PrivateRoot"
    private_root.mkdir()
    case_alias = tmp_path / "privateroot"
    if not case_alias.exists():
        pytest.skip("case-insensitive path aliases are unavailable")
    monkeypatch.setenv("LTX_LORA_PRIVATE_ROOT", str(case_alias))

    with pytest.raises(ValueError, match="approved private root") as exc_info:
        approved_private_root_from_environment()

    assert str(case_alias) not in str(exc_info.value)


def test_ledger_resolution_is_absolute_and_independent_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    ledger_path = (
        private_root / "pilots" / PILOT_ID / "ledger" / "pilot.sqlite3"
    )
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(b"synthetic ledger placeholder")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    assert resolve_pilot_ledger(private_root, PILOT_ID) == ledger_path.resolve(
        strict=True
    )


def test_run_resolution_requires_exact_canonical_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    run_dir = (
        private_root / "pilots" / PILOT_ID / "runs" / EXECUTION_ID
    )
    run_dir.mkdir(parents=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    assert require_canonical_run_dir(
        private_root,
        PILOT_ID,
        EXECUTION_ID,
        run_dir,
    ) == run_dir.resolve(strict=True)


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "..",
        "pilot_../escape",
        "PILOT_00000000000040008000000000000001",
        "pilot_00000000000010008000000000000001",
        "pilot_00000000000040007000000000000001",
        "pilot_00000000000040008000000000000001 ",
    ],
)
def test_ledger_resolution_rejects_unsafe_pilot_ids_neutrally(
    tmp_path: Path,
    unsafe_id: str,
) -> None:
    private_root = tmp_path / "private-root"
    private_root.mkdir()

    with pytest.raises(ValueError, match="pilot_id is invalid") as exc_info:
        resolve_pilot_ledger(private_root, unsafe_id)

    assert unsafe_id not in str(exc_info.value)


def test_ledger_resolution_rejects_missing_or_aliased_layout_neutrally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    ledger_path = (
        private_root / "pilots" / PILOT_ID / "ledger" / "pilot.sqlite3"
    )
    ledger_path.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="private workspace path") as exc_info:
        resolve_pilot_ledger(private_root, PILOT_ID)
    assert str(ledger_path) not in str(exc_info.value)

    ledger_path.write_bytes(b"synthetic ledger placeholder")
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == ledger_path.parent or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValueError, match="private workspace path") as alias_exc:
        resolve_pilot_ledger(private_root, PILOT_ID)
    assert str(ledger_path) not in str(alias_exc.value)


def test_ledger_resolution_rejects_case_aliased_layout_neutrally(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private-root"
    actual = (
        private_root / "PILOTS" / PILOT_ID / "ledger" / "pilot.sqlite3"
    )
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"synthetic ledger placeholder")
    expected_alias = (
        private_root / "pilots" / PILOT_ID / "ledger" / "pilot.sqlite3"
    )
    if not expected_alias.exists():
        pytest.skip("case-insensitive path aliases are unavailable")

    with pytest.raises(ValueError, match="private workspace path") as exc_info:
        resolve_pilot_ledger(private_root, PILOT_ID)

    assert str(expected_alias) not in str(exc_info.value)


def test_run_resolution_rejects_mismatch_relative_and_alias_neutrally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private-root"
    expected = private_root / "pilots" / PILOT_ID / "runs" / EXECUTION_ID
    expected.mkdir(parents=True)
    mismatch = private_root / "pilots" / PILOT_ID / "runs" / "other"
    mismatch.mkdir()

    for supplied in (mismatch, Path("relative-run")):
        with pytest.raises(ValueError) as exc_info:
            require_canonical_run_dir(
                private_root,
                PILOT_ID,
                EXECUTION_ID,
                supplied,
            )
        assert str(supplied) not in str(exc_info.value)

    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == expected or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValueError, match="private workspace path") as alias_exc:
        require_canonical_run_dir(
            private_root,
            PILOT_ID,
            EXECUTION_ID,
            expected,
        )
    assert str(expected) not in str(alias_exc.value)


@pytest.mark.parametrize(
    "unsafe_id",
    ["..", "exec_../escape", "EXEC_00000000000040008000000000000002"],
)
def test_run_resolution_rejects_unsafe_execution_ids_neutrally(
    tmp_path: Path,
    unsafe_id: str,
) -> None:
    private_root = tmp_path / "private-root"
    private_root.mkdir()

    with pytest.raises(ValueError, match="execution_id is invalid") as exc_info:
        require_canonical_run_dir(
            private_root,
            PILOT_ID,
            unsafe_id,
            private_root,
        )

    assert unsafe_id not in str(exc_info.value)
