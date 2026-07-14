from pathlib import Path

import pytest

from ltx_lora_pilot.dataset import safe_reset_output_directory


def test_safe_reset_accepts_named_child(tmp_path: Path) -> None:
    output_root = tmp_path / "dataset"
    training = output_root / "training"
    training.mkdir(parents=True)
    (training / "old.txt").write_text("old", encoding="utf-8")

    safe_reset_output_directory(output_root, training)

    assert training.is_dir()
    assert list(training.iterdir()) == []


@pytest.mark.parametrize("candidate_name", ["archive", "other", "private_manifest.json"])
def test_safe_reset_rejects_unapproved_child(tmp_path: Path, candidate_name: str) -> None:
    output_root = tmp_path / "dataset"
    candidate = output_root / candidate_name

    with pytest.raises(ValueError, match="refusing to reset"):
        safe_reset_output_directory(output_root, candidate)


def test_safe_reset_rejects_path_outside_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "dataset"
    outside = tmp_path / "training"

    with pytest.raises(ValueError, match="refusing to reset"):
        safe_reset_output_directory(output_root, outside)
