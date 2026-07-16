from __future__ import annotations

import os
from pathlib import Path

import pytest

import ltx_lora_pilot.staging as staging
from ltx_lora_pilot.staging import StagedArtifactChanged, stage_bundle
from test_preflight import EXECUTION_ID, PILOT_ID, _write_ready_run


@pytest.fixture
def ready_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    fixture = _write_ready_run(tmp_path)
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda _path: None)
    return fixture


def _stage(fixture: dict[str, object]):
    return stage_bundle(
        fixture["run_dir"],  # type: ignore[arg-type]
        approved_private_root=fixture["private_root"],  # type: ignore[arg-type]
        confirmed_bundle_id=fixture["bundle_id"],  # type: ignore[arg-type]
        pilot_id=PILOT_ID,
        execution_id=EXECUTION_ID,
    )


def test_stage_bundle_copies_exact_archive_and_two_bound_validation_pairs(
    ready_run: dict[str, object],
) -> None:
    source_archive = Path(ready_run["run_dir"]) / "bundle" / "training-data.zip"
    source_size = source_archive.stat().st_size

    with _stage(ready_run) as staged:
        session = staged._stage_session
        assert staged.training_zip.parent.name == staged.bundle_id
        assert staged.training_zip.stat().st_size == source_size
        assert len(staged.validation_pairs) == 2
        assert {pair.group_id for pair in staged.validation_pairs}
        assert all(pair.image.is_file() and pair.audio.is_file() for pair in staged.validation_pairs)
        source_archive.rename(source_archive.with_name("source-renamed.zip"))
        assert staged.verify_unchanged() is True

    assert not session.exists()


def test_stage_bundle_detects_replaced_or_modified_staged_file(
    ready_run: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoopGuard:
        def close(self) -> None:
            pass

    monkeypatch.setattr(staging, "_open_platform_read_guard", lambda _path: NoopGuard())
    with _stage(ready_run) as staged:
        os.chmod(staged.training_zip, 0o600)
        staged.training_zip.write_bytes(staged.training_zip.read_bytes() + b"x")

        assert staged.verify_unchanged() is False
        with pytest.raises(StagedArtifactChanged, match="staged artifact changed"):
            staged.require_unchanged()


def test_staged_execution_config_cannot_be_mutated_in_memory(
    ready_run: dict[str, object],
) -> None:
    with _stage(ready_run) as staged:
        with pytest.raises(TypeError):
            staged.execution_config["steps"] = 777_777  # type: ignore[index]

        assert staged.execution_config["steps"] == 1000


def test_partial_platform_guard_acquisition_closes_prior_handles(
    ready_run: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    guards = []

    class Guard:
        closed = False

        def close(self) -> None:
            self.closed = True

    def acquire(_path: Path) -> Guard:
        if len(guards) == 2:
            raise RuntimeError("synthetic guard acquisition failure")
        guard = Guard()
        guards.append(guard)
        return guard

    monkeypatch.setattr(staging, "_open_platform_read_guard", acquire)

    with pytest.raises(RuntimeError, match="synthetic guard"):
        with _stage(ready_run):
            pass

    assert len(guards) == 2
    assert all(guard.closed for guard in guards)
    stage_root = Path(ready_run["private_root"]) / ".a2v-staging"
    assert not list(stage_root.iterdir())
