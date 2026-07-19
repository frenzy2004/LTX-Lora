from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from ltx_lora_pilot.artifacts import canonical_json_bytes
from ltx_lora_pilot.provider_validation import (
    build_provider_validation_selection,
    validate_provider_validation_selection,
)


def _group_id(index: int) -> str:
    return f"grp_000000000000400080000000{index:08x}"


def _digest(name: str, content: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    groups = []
    for index in range(1, 17):
        group_id = _group_id(index)
        files = []
        for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
            name = f"{group_id}{suffix}"
            content = f"opaque-{index}:{suffix}".encode("ascii")
            (candidate_dir / name).write_bytes(content)
            files.append(_digest(name, content))
        groups.append({"group_id": group_id, "files": files})

    structural_report = {
        "schema_version": "a2v-structural-report-v1",
        "status": "valid",
        "spec": {
            "width": 544,
            "height": 960,
            "frames": 89,
            "fps": 24,
            "sample_rate": 48_000,
        },
        "groups": list(reversed(groups)),
    }
    train_ids = [_group_id(index) for index in range(1, 11)]
    holdout_ids = [_group_id(index) for index in range(11, 16)]
    quality_summary = {
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
            "training_teeth_or_inner_mouth_visible": 3,
            "holdout_teeth_or_inner_mouth_visible": 2,
        },
    }
    execution_config = {
        "schema_version": "a2v-execution-config-v2",
        "canonical_json_version": 1,
        "execution_id": "exec_00000000000040008000000000000001",
        "pilot_id": "pilot_00000000000040008000000000000002",
        "ledger_id": "ledger_00000000000040008000000000000003",
        "created_at_utc": "2026-07-15T01:00:00Z",
        "expires_at_utc": "2026-07-15T20:00:00Z",
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "trigger_phrase": "chrx9_speech",
        "rank": 32,
        "steps": 1_000,
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
        "negative_prompt": "synthetic artifacts, distortion",
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
        "dataset_manifest_sha256": "1" * 64,
        "training_archive_sha256": "2" * 64,
        "standing_authorization_sha256": "3" * 64,
        "price_evidence_sha256": "4" * 64,
        "price_source_url": "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "rate_usd_per_step": "0.006",
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
    }
    prompts = {
        _group_id(12): "A close talking-head shot with natural speech and subtle facial motion.",
        _group_id(11): "A medium talking-head shot with natural speech and steady eye contact.",
    }
    return {
        "candidate_dir": candidate_dir,
        "structural_report": structural_report,
        "quality_summary": quality_summary,
        "execution_config": execution_config,
        "prompts": prompts,
    }


def _build(fixture: dict[str, Any]) -> dict[str, Any]:
    return build_provider_validation_selection(
        structural_report=fixture["structural_report"],
        quality_summary=fixture["quality_summary"],
        execution_config=fixture["execution_config"],
        candidate_dir=fixture["candidate_dir"],
        prompts=fixture["prompts"],
    )


def _validate(selection: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    return validate_provider_validation_selection(
        selection,
        fixture["structural_report"],
        fixture["quality_summary"],
        fixture["execution_config"],
        fixture["candidate_dir"],
    )


def test_builder_produces_exact_canonical_selection_bound_to_current_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    selection = _build(fixture)

    assert set(selection) == {
        "schema_version",
        "canonical_json_version",
        "structural_report_sha256",
        "execution_config_sha256",
        "items",
    }
    assert selection["schema_version"] == "a2v-provider-validation-selection-v1"
    assert selection["canonical_json_version"] == 1
    assert selection["structural_report_sha256"] == hashlib.sha256(
        canonical_json_bytes(fixture["structural_report"])
    ).hexdigest()
    assert selection["execution_config_sha256"] == hashlib.sha256(
        canonical_json_bytes(fixture["execution_config"])
    ).hexdigest()
    assert [item["group_id"] for item in selection["items"]] == [
        _group_id(11),
        _group_id(12),
    ]
    assert all(set(item) == {"group_id", "prompt", "image", "audio"} for item in selection["items"])
    assert all(set(item["image"]) == {"name", "bytes", "sha256"} for item in selection["items"])
    assert all(set(item["audio"]) == {"name", "bytes", "sha256"} for item in selection["items"])
    assert _validate(selection, fixture) == selection


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("top_unknown", "selection must contain the exact fields"),
        ("item_unknown", "selection item 0 must contain the exact fields"),
        ("digest_unknown", "selection item 0 image must contain the exact fields"),
        ("schema", "selection schema mismatch"),
        ("canonical_version", "canonical JSON version mismatch"),
        ("one_item", "exactly two items"),
        ("three_items", "exactly two items"),
        ("reverse", "canonical group-ID order"),
        ("duplicate_group", "distinct holdout groups"),
        ("train_group", "accepted holdout group"),
        ("rejected_group", "accepted holdout group"),
        ("missing_group", "missing structural group"),
        ("stale_structural", "structural report digest mismatch"),
        ("stale_config", "execution config digest mismatch"),
        ("url_alias", "canonical local filename"),
        ("path_separator", "canonical local filename"),
        ("bool_bytes", "positive integer"),
        ("uppercase_hash", "lowercase SHA-256"),
    ],
)
def test_validator_rejects_noncanonical_or_unbound_selection(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    if mutation == "top_unknown":
        selection["validation"] = []
    elif mutation == "item_unknown":
        selection["items"][0]["image_url"] = "https://example.invalid/private"
    elif mutation == "digest_unknown":
        selection["items"][0]["image"]["path"] = "private.png"
    elif mutation == "schema":
        selection["schema_version"] = "a2v-provider-validation-selection-v2"
    elif mutation == "canonical_version":
        selection["canonical_json_version"] = True
    elif mutation == "one_item":
        selection["items"].pop()
    elif mutation == "three_items":
        selection["items"].append(copy.deepcopy(selection["items"][1]))
    elif mutation == "reverse":
        selection["items"].reverse()
    elif mutation == "duplicate_group":
        selection["items"][1] = copy.deepcopy(selection["items"][0])
    elif mutation == "train_group":
        selection["items"][0]["group_id"] = _group_id(1)
    elif mutation == "rejected_group":
        selection["items"][0]["group_id"] = _group_id(16)
    elif mutation == "missing_group":
        selection["items"][0]["group_id"] = _group_id(99)
    elif mutation == "stale_structural":
        selection["structural_report_sha256"] = "0" * 64
    elif mutation == "stale_config":
        selection["execution_config_sha256"] = "0" * 64
    elif mutation == "url_alias":
        selection["items"][0]["image"]["name"] = "https://example.invalid/start.png"
    elif mutation == "path_separator":
        selection["items"][0]["audio"]["name"] = "nested/audio.wav"
    elif mutation == "bool_bytes":
        selection["items"][0]["image"]["bytes"] = True
    else:
        selection["items"][0]["audio"]["sha256"] = "A" * 64

    with pytest.raises(ValueError, match=message):
        _validate(selection, fixture)


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ("", "canonical prompt"),
        (" trailing space ", "canonical prompt"),
        ("e\N{COMBINING ACUTE ACCENT}", "NFC-normalized"),
        ("x" * 1_025, "at most 1024 UTF-8 bytes"),
        ("line one\nline two", "prohibited Unicode character"),
        ("joined\N{ZERO WIDTH JOINER}text", "prohibited Unicode character"),
        ("surrogate\ud800", "prohibited Unicode character"),
        ("private\ue000", "prohibited Unicode character"),
        ("see https://example.invalid", "must not contain a URL"),
    ],
)
def test_builder_rejects_noncanonical_prompt(
    tmp_path: Path,
    prompt: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["prompts"][_group_id(11)] = prompt

    with pytest.raises(ValueError, match=message):
        _build(fixture)


def test_validator_rejects_selection_record_that_differs_from_structural_record(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    selection["items"][0]["image"]["bytes"] += 1

    with pytest.raises(ValueError, match="image structural record mismatch"):
        _validate(selection, fixture)


def test_validator_freshly_hashes_candidate_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    image_name = selection["items"][0]["image"]["name"]
    (fixture["candidate_dir"] / image_name).write_bytes(b"changed-after-review")

    with pytest.raises(ValueError, match="current candidate bytes mismatch"):
        _validate(selection, fixture)


def test_builder_rejects_duplicate_selected_media(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first_name = f"{_group_id(11)}_start.png"
    second_name = f"{_group_id(12)}_start.png"
    duplicate = (fixture["candidate_dir"] / first_name).read_bytes()
    (fixture["candidate_dir"] / second_name).write_bytes(duplicate)
    second_group = next(
        group
        for group in fixture["structural_report"]["groups"]
        if group["group_id"] == _group_id(12)
    )
    second_image = next(record for record in second_group["files"] if record["name"] == second_name)
    second_image.update(_digest(second_name, duplicate))

    with pytest.raises(ValueError, match="duplicate selected media"):
        _build(fixture)


def test_validator_rejects_linked_or_aliased_candidate_files(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    image_name = selection["items"][0]["image"]["name"]
    image_path = fixture["candidate_dir"] / image_name
    target = tmp_path / "outside.png"
    target.write_bytes(image_path.read_bytes())
    image_path.unlink()
    try:
        image_path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="must not be a link or alias"):
        _validate(selection, fixture)


def test_validator_rejects_hardlink_alias_between_selected_files(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    first_name = f"{_group_id(11)}_audio.wav"
    second_name = f"{_group_id(12)}_audio.wav"
    first_path = fixture["candidate_dir"] / first_name
    second_path = fixture["candidate_dir"] / second_name
    second_path.unlink()
    try:
        os.link(first_path, second_path)
    except OSError:
        pytest.skip("hardlink creation is unavailable")
    duplicate = first_path.read_bytes()
    second_group = next(
        group
        for group in fixture["structural_report"]["groups"]
        if group["group_id"] == _group_id(12)
    )
    second_audio = next(record for record in second_group["files"] if record["name"] == second_name)
    second_audio.update(_digest(second_name, duplicate))
    selection["items"][1]["audio"] = copy.deepcopy(second_audio)
    selection["structural_report_sha256"] = hashlib.sha256(
        canonical_json_bytes(fixture["structural_report"])
    ).hexdigest()

    with pytest.raises(ValueError, match="must not be a link or alias"):
        _validate(selection, fixture)


def test_validator_rejects_stale_structural_or_config_objects_even_with_old_digests(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    fixture["execution_config"]["created_at_utc"] = "2026-07-15T01:00:01Z"

    with pytest.raises(ValueError, match="execution config digest mismatch"):
        _validate(selection, fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "execution configuration must contain the exact fields"),
        ("unknown", "execution configuration must contain the exact fields"),
        ("endpoint", "endpoint mismatch"),
        ("resolution", "execution configuration resolution mismatch"),
        (
            "validation_frames",
            "execution configuration validation number of frames mismatch",
        ),
        (
            "validation_resolution",
            "execution configuration validation resolution mismatch",
        ),
    ],
)
def test_validator_rejects_digest_consistent_malformed_execution_config(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    config = fixture["execution_config"]
    if mutation == "missing":
        del config["rank"]
    elif mutation == "unknown":
        config["validation"] = []
    elif mutation == "endpoint":
        config["endpoint"] = "fal-ai/ltx23-trainer-v2/i2v"
    elif mutation == "resolution":
        config["resolution"] = "low"
    elif mutation == "validation_frames":
        config["validation_number_of_frames"] = 88
    else:
        config["validation_resolution"] = "low"
    selection["execution_config_sha256"] = hashlib.sha256(
        canonical_json_bytes(config)
    ).hexdigest()

    with pytest.raises(ValueError, match=message):
        _validate(selection, fixture)


def test_validator_rejects_selected_file_with_unselected_external_hardlink(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    selected_name = selection["items"][0]["image"]["name"]
    selected_path = fixture["candidate_dir"] / selected_name
    external_alias = tmp_path / "unselected-external-alias.png"
    try:
        os.link(selected_path, external_alias)
    except OSError:
        pytest.skip("hardlink creation is unavailable")

    with pytest.raises(ValueError, match="must not be a link or alias"):
        _validate(selection, fixture)
