from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest

from ltx_lora_pilot.a2v_quality import (
    load_quality_attestation,
    validate_quality_and_splits,
)


REQUIRED_CHECKS = (
    "one_visible_speaker",
    "close_or_medium_close_framing",
    "face_mouth_jaw_cheeks_and_eyes_unobstructed",
    "continuous_real_speech_motion",
    "no_internal_cut",
    "no_overlapping_speaker_dubbing_or_music",
    "no_watermark_burned_captions_or_beauty_filter",
    "audio_and_video_are_from_the_same_interval",
    "rights_and_likeness_use_confirmed",
)
TEETH_CHECK = "teeth_or_inner_mouth_visible"

ROOT = Path(__file__).resolve().parents[1]
CLI_SPEC = spec_from_file_location(
    "validate_a2v_dataset",
    ROOT / "scripts" / "validate_a2v_dataset.py",
)
assert CLI_SPEC and CLI_SPEC.loader
CLI_MODULE = module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI_MODULE)


def _group_id(index: int) -> str:
    return f"sample_{index + 1:03d}"


def make_structural_report(count: int) -> dict:
    groups = []
    for index in range(count):
        group_id = _group_id(index)
        names = sorted(
            [
                f"{group_id}_start.png",
                f"{group_id}_audio.wav",
                f"{group_id}_end.mp4",
                f"{group_id}.txt",
            ]
        )
        files = [
            {
                "name": name,
                "bytes": index + offset + 1,
                "sha256": hashlib.sha256(f"{group_id}:{name}".encode()).hexdigest(),
            }
            for offset, name in enumerate(names)
        ]
        groups.append({"group_id": group_id, "files": files})
    return {
        "schema_version": "a2v-structural-report-v1",
        "status": "valid",
        "spec": {
            "width": 544,
            "height": 960,
            "frames": 89,
            "fps": 24,
            "sample_rate": 48_000,
        },
        "groups": groups,
    }


def make_attestation(
    *,
    train: int = 10,
    holdout: int = 5,
    shared_session: bool = False,
) -> dict:
    groups = []
    for index in range(train + holdout):
        is_train = index < train
        checks = {name: True for name in REQUIRED_CHECKS}
        checks[TEETH_CHECK] = not is_train and index == train
        groups.append(
            {
                "group_id": _group_id(index),
                "split": "train" if is_train else "holdout",
                "accepted": True,
                "source_asset_id": f"asset-{index + 1:03d}",
                "source_session_id": f"session-{index + 1:03d}",
                "location_id": (
                    f"training-location-{(index % 3) + 1}"
                    if is_train
                    else f"holdout-location-{index - train + 1}"
                ),
                "source_start_ms": index * 5_000,
                "source_end_ms": index * 5_000 + 3_708,
                "checks": checks,
                "notes": "Accepted after private review.",
            }
        )
    if shared_session:
        groups[train]["source_session_id"] = groups[0]["source_session_id"]
    return {
        "schema_version": "a2v-quality-attestation-v1",
        "dataset_id": "character-speech-pilot",
        "rights_and_consent": {
            "confirmed": True,
            "reviewer_id": "operator-1",
            "reviewed_at_utc": "2026-07-15T00:00:00Z",
        },
        "groups": groups,
    }


def test_load_quality_attestation_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    path.write_text(
        '{"schema_version":"a2v-quality-attestation-v1","groups":[],"groups":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_quality_attestation(path)


def test_quality_returns_only_neutral_ids_and_counts() -> None:
    result = validate_quality_and_splits(make_attestation(), make_structural_report(15))

    train_ids = [_group_id(index) for index in range(10)]
    holdout_ids = [_group_id(index) for index in range(10, 15)]
    assert result == {
        "status": "valid",
        "accepted_train_group_ids": train_ids,
        "accepted_holdout_group_ids": holdout_ids,
        "location_coverage": {
            "isolated_holdout_group_ids": holdout_ids,
        },
        "coverage_counts": {
            "accepted_train_groups": 10,
            "accepted_holdout_groups": 5,
            "location_isolated_holdout_groups": 5,
            "training_teeth_or_inner_mouth_visible": 0,
            "holdout_teeth_or_inner_mouth_visible": 1,
        },
    }


def test_quality_requires_ten_train_and_five_holdout() -> None:
    attestation = make_attestation(train=9, holdout=5)

    with pytest.raises(ValueError, match="at least 10 accepted training groups"):
        validate_quality_and_splits(attestation, make_structural_report(14))


def test_quality_requires_five_holdouts() -> None:
    attestation = make_attestation(train=10, holdout=4)

    with pytest.raises(ValueError, match="at least 5 accepted holdout groups"):
        validate_quality_and_splits(attestation, make_structural_report(14))


def test_quality_rejects_session_crossing_splits() -> None:
    attestation = make_attestation(train=10, holdout=5, shared_session=True)

    with pytest.raises(ValueError, match="source session crosses"):
        validate_quality_and_splits(attestation, make_structural_report(15))


@pytest.mark.parametrize(
    "field",
    ["source_asset_id", "source_session_id", "location_id"],
)
def test_quality_rejects_noncanonical_provenance_id(field: str) -> None:
    attestation = make_attestation()
    attestation["groups"][10][field] += " "

    with pytest.raises(ValueError, match="canonical opaque ID"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_false_rights_confirmation() -> None:
    attestation = make_attestation()
    attestation["rights_and_consent"]["confirmed"] = False

    with pytest.raises(ValueError, match="rights and consent must be confirmed"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_nonstring_split_as_validation_error() -> None:
    attestation = make_attestation()
    attestation["groups"][0]["split"] = []

    with pytest.raises(ValueError, match="split must be train or holdout"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_missing_required_check() -> None:
    attestation = make_attestation()
    del attestation["groups"][0]["checks"]["one_visible_speaker"]

    with pytest.raises(ValueError, match="missing required check"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_accepted_group_with_false_required_check() -> None:
    attestation = make_attestation()
    attestation["groups"][0]["checks"]["continuous_real_speech_motion"] = False

    with pytest.raises(ValueError, match="accepted group has a false required check"):
        validate_quality_and_splits(attestation, make_structural_report(15))


@pytest.mark.parametrize("object_name", ["top", "rights", "group", "checks"])
def test_quality_rejects_unknown_keys_in_every_attestation_object(object_name: str) -> None:
    attestation = make_attestation()
    if object_name == "top":
        target = attestation
    elif object_name == "rights":
        target = attestation["rights_and_consent"]
    elif object_name == "group":
        target = attestation["groups"][0]
    else:
        target = attestation["groups"][0]["checks"]
    target["unexpected"] = True

    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_duplicate_group_id() -> None:
    attestation = make_attestation()
    attestation["groups"][1]["group_id"] = attestation["groups"][0]["group_id"]

    with pytest.raises(ValueError, match="duplicate group ID"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_group_missing_from_structural_report() -> None:
    attestation = make_attestation()

    with pytest.raises(ValueError, match="missing structural group"):
        validate_quality_and_splits(attestation, make_structural_report(14))


def test_quality_rejects_structural_group_missing_from_attestation() -> None:
    attestation = make_attestation()
    attestation["groups"].pop()

    with pytest.raises(ValueError, match="missing quality attestation group"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_overlapping_source_interval_across_splits() -> None:
    attestation = make_attestation()
    train_group = attestation["groups"][0]
    holdout_group = attestation["groups"][10]
    holdout_group["source_asset_id"] = train_group["source_asset_id"]
    holdout_group["source_start_ms"] = train_group["source_end_ms"] - 1
    holdout_group["source_end_ms"] = train_group["source_end_ms"] + 1_000

    with pytest.raises(ValueError, match="source interval overlaps"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_duplicate_media_digest() -> None:
    structural_report = make_structural_report(15)
    first_media = next(
        item
        for item in structural_report["groups"][0]["files"]
        if item["name"].endswith("_end.mp4")
    )
    second_media = next(
        item
        for item in structural_report["groups"][1]["files"]
        if item["name"].endswith("_end.mp4")
    )
    second_media["sha256"] = first_media["sha256"]

    with pytest.raises(ValueError, match="duplicate media digest"):
        validate_quality_and_splits(make_attestation(), structural_report)


def test_quality_rejects_noninteger_structural_spec_value() -> None:
    structural_report = make_structural_report(15)
    structural_report["spec"]["width"] = 544.0

    with pytest.raises(ValueError, match="exact normalized A2V spec"):
        validate_quality_and_splits(make_attestation(), structural_report)


def test_quality_requires_two_location_isolated_holdouts() -> None:
    attestation = make_attestation()
    training_location = attestation["groups"][0]["location_id"]
    for group in attestation["groups"][10:14]:
        group["location_id"] = training_location

    with pytest.raises(ValueError, match="at least two location-isolated holdouts"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_requires_heldout_teeth_or_inner_mouth_coverage() -> None:
    attestation = make_attestation()
    for group in attestation["groups"]:
        group["checks"][TEETH_CHECK] = False

    with pytest.raises(ValueError, match="held-out teeth or inner-mouth coverage"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_rejects_malformed_source_interval() -> None:
    attestation = make_attestation()
    attestation["groups"][0]["source_end_ms"] = attestation["groups"][0]["source_start_ms"]

    with pytest.raises(ValueError, match="source interval"):
        validate_quality_and_splits(attestation, make_structural_report(15))


def test_quality_excludes_rejected_groups_from_accepted_outputs() -> None:
    attestation = make_attestation(train=11, holdout=5)
    rejected = attestation["groups"][0]
    rejected["accepted"] = False
    rejected["notes"] = "Rejected after private review."
    rejected["checks"]["continuous_real_speech_motion"] = False

    result = validate_quality_and_splits(attestation, make_structural_report(16))

    assert rejected["group_id"] not in result["accepted_train_group_ids"]
    assert result["coverage_counts"]["accepted_train_groups"] == 10
    assert "notes" not in result


def test_validation_cli_requires_attestation_and_structural_report_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_a2v_dataset.py", "--dataset-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit):
        CLI_MODULE.main()

    error = capsys.readouterr().err
    assert "--quality-attestation" in error
    assert "--structural-report" in error


def test_validation_cli_writes_structural_report_and_prints_sanitized_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    attestation_path = tmp_path / "private-attestation.json"
    structural_path = tmp_path / "private-structural-report.json"
    structural_report = make_structural_report(15)
    attestation = make_attestation()
    quality_result = {
        "status": "valid",
        "accepted_train_group_ids": [_group_id(index) for index in range(10)],
        "accepted_holdout_group_ids": [_group_id(index) for index in range(10, 15)],
        "location_coverage": {
            "isolated_holdout_group_ids": [_group_id(index) for index in range(10, 15)],
        },
        "coverage_counts": {
            "accepted_train_groups": 10,
            "accepted_holdout_groups": 5,
            "location_isolated_holdout_groups": 5,
            "training_teeth_or_inner_mouth_visible": 0,
            "holdout_teeth_or_inner_mouth_visible": 1,
        },
    }

    def validate_directory(root: Path, *, spec: object, trigger_phrase: str | None) -> dict:
        assert root == dataset_dir
        assert trigger_phrase == "managed-trigger"
        return structural_report

    def load_attestation(path: Path) -> dict:
        assert path == attestation_path
        return attestation

    def validate_quality(loaded: dict, structural: dict) -> dict:
        assert loaded is attestation
        assert structural is structural_report
        return quality_result

    monkeypatch.setattr(CLI_MODULE, "validate_a2v_directory", validate_directory)
    monkeypatch.setattr(
        CLI_MODULE,
        "load_quality_attestation",
        load_attestation,
        raising=False,
    )
    monkeypatch.setattr(
        CLI_MODULE,
        "validate_quality_and_splits",
        validate_quality,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_a2v_dataset.py",
            "--dataset-dir",
            str(dataset_dir),
            "--quality-attestation",
            str(attestation_path),
            "--structural-report",
            str(structural_path),
            "--trigger-phrase",
            "managed-trigger",
        ],
    )

    CLI_MODULE.main()

    output = capsys.readouterr().out
    assert json.loads(output) == quality_result
    assert "private-attestation" not in output
    assert "operator-1" not in output
    assert "asset-001" not in output
    assert "Accepted after private review" not in output
    assert json.loads(structural_path.read_text(encoding="utf-8")) == structural_report


@pytest.mark.parametrize("destination", ["attestation", "dataset"])
def test_validation_cli_rejects_structural_report_input_alias(
    destination: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    attestation_path = tmp_path / "quality-attestation.json"
    attestation_path.write_text("private attestation sentinel", encoding="utf-8")
    if destination == "attestation":
        structural_path = attestation_path
    else:
        structural_path = dataset_dir / "sample_001.txt"
        structural_path.write_text("candidate sentinel", encoding="utf-8")
    original_bytes = structural_path.read_bytes()

    monkeypatch.setattr(
        CLI_MODULE,
        "validate_a2v_directory",
        lambda *args, **kwargs: make_structural_report(15),
    )
    monkeypatch.setattr(
        CLI_MODULE,
        "load_quality_attestation",
        lambda path: make_attestation(),
    )
    monkeypatch.setattr(
        CLI_MODULE,
        "validate_quality_and_splits",
        lambda *args: {"status": "valid"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_a2v_dataset.py",
            "--dataset-dir",
            str(dataset_dir),
            "--quality-attestation",
            str(attestation_path),
            "--structural-report",
            str(structural_path),
        ],
    )

    with pytest.raises(SystemExit):
        CLI_MODULE.main()

    captured = capsys.readouterr()
    assert "A2V_VALIDATION_FAILED" in captured.err
    assert captured.out == ""
    assert structural_path.read_bytes() == original_bytes


def test_validation_cli_sanitizes_private_paths_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "private-source-location"
    attestation_path = tmp_path / "private-attestation.json"
    structural_path = tmp_path / "structural-report.json"

    def fail_validation(*args: object, **kwargs: object) -> dict:
        raise NotADirectoryError(str(dataset_dir))

    monkeypatch.setattr(CLI_MODULE, "validate_a2v_directory", fail_validation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_a2v_dataset.py",
            "--dataset-dir",
            str(dataset_dir),
            "--quality-attestation",
            str(attestation_path),
            "--structural-report",
            str(structural_path),
        ],
    )

    with pytest.raises(SystemExit):
        CLI_MODULE.main()

    captured = capsys.readouterr()
    assert "A2V_VALIDATION_FAILED" in captured.err
    assert str(dataset_dir) not in captured.err
    assert str(attestation_path) not in captured.err
    assert str(structural_path) not in captured.err
    assert captured.out == ""
    assert not structural_path.exists()
