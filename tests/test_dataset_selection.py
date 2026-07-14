import pytest

from ltx_lora_pilot.dataset import portrait_video_filter, select_records


def _record(source_id: str, duration: float = 8.0) -> dict:
    return {
        "source_id": source_id,
        "duration": duration,
        "width": 3840,
        "height": 2160,
        "path": f"private/{source_id}.mp4",
    }


def test_select_records_preserves_explicit_order() -> None:
    records = [_record("a"), _record("b"), _record("c")]

    training, holdout = select_records(records, ["b", "a"], ["c"], clip_seconds=5.0)

    assert [record["source_id"] for record in training] == ["b", "a"]
    assert [record["source_id"] for record in holdout] == ["c"]


def test_select_records_rejects_overlap() -> None:
    records = [_record("a"), _record("b")]

    with pytest.raises(ValueError, match="overlap"):
        select_records(records, ["a"], ["a"], clip_seconds=5.0)


def test_select_records_rejects_missing_or_short_sources() -> None:
    records = [_record("a"), _record("short", duration=4.0)]

    with pytest.raises(ValueError, match="missing or ineligible"):
        select_records(records, ["a", "short"], ["unknown"], clip_seconds=5.0)


def test_portrait_filter_crops_center_and_emits_720p_portrait() -> None:
    video_filter = portrait_video_filter()

    assert "crop=" in video_filter
    assert "scale=720:1280" in video_filter
    assert video_filter.startswith("fps=24,")
