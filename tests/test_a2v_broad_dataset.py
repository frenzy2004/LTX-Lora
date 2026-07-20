from __future__ import annotations

import json
import sys
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))


def _write_approved_manifest(path: Path, source_count: int = 20) -> None:
    path.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "source_id": f"A{index:03d}",
                        "source_relative_path": f"source-{index:03d}.mov",
                        "window_index": 1,
                        "start": 1.0,
                        "end": 1.0 + 89 / 24,
                        "word_count": 12,
                        "speech_seconds": 3.0,
                        "crop": {"x": 0, "y": 0, "width": 544, "height": 960},
                        "caption": "[SPEECH] A person speaks naturally to the camera.",
                        "visual_status": "accepted",
                    }
                    for index in range(1, source_count + 1)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_split_sources_is_deterministic_and_disjoint() -> None:
    from a2v_broad_dataset import split_sources

    source_ids = [f"src-{index:03d}" for index in range(50)]

    train, holdout = split_sources(
        source_ids,
        holdout_fraction=0.10,
        min_holdout=5,
        seed=42,
    )

    assert len(train) == 45
    assert len(holdout) == 5
    assert set(train).isdisjoint(holdout)
    assert sorted(train + holdout) == source_ids
    assert (train, holdout) == split_sources(
        list(reversed(source_ids)),
        holdout_fraction=0.10,
        min_holdout=5,
        seed=42,
    )


def test_select_speech_windows_returns_exact_non_overlapping_intervals() -> None:
    from a2v_broad_dataset import Word, select_speech_windows

    words = [
        Word(start=offset + index * 0.42, end=offset + index * 0.42 + 0.28, text=f"w{index}")
        for offset in (0.5, 8.5)
        for index in range(10)
    ]

    windows = select_speech_windows(
        words,
        source_duration=16.0,
        clip_seconds=89 / 24,
        max_windows=2,
    )

    assert len(windows) == 2
    assert all(abs(window.duration - 89 / 24) < 1e-9 for window in windows)
    assert all(window.word_count >= 8 for window in windows)
    assert windows[0].end <= windows[1].start


def test_crop_tracks_off_centre_face_and_contains_temporal_envelope() -> None:
    from a2v_broad_dataset import Box, FaceObservation, derive_portrait_crop

    observations = [
        FaceObservation(0.0, primary=Box(2800, 500, 420, 500)),
        FaceObservation(1.0, primary=Box(2920, 520, 430, 500)),
        FaceObservation(2.0, primary=Box(2850, 510, 425, 510)),
    ]

    decision = derive_portrait_crop((3840, 2160), observations)

    assert decision.accepted
    assert decision.crop is not None
    assert decision.crop.width * 30 == decision.crop.height * 17
    assert decision.crop.x > (3840 - decision.crop.width) / 2
    assert all(decision.crop.contains(item.primary) for item in observations)


def test_crop_preserves_exact_bucket_aspect_for_small_proxy_faces() -> None:
    from a2v_broad_dataset import Box, FaceObservation, derive_portrait_crop

    observations = [
        FaceObservation(0.0, primary=Box(240, 420, 50, 62)),
        FaceObservation(1.0, primary=Box(242, 418, 52, 62)),
        FaceObservation(2.0, primary=Box(238, 421, 50, 63)),
    ]

    decision = derive_portrait_crop((540, 960), observations)

    assert decision.accepted
    assert decision.crop is not None
    assert decision.crop.width * 30 == decision.crop.height * 17
    assert all(decision.crop.contains(item.primary) for item in observations)


def test_crop_rejects_prominent_second_face() -> None:
    from a2v_broad_dataset import Box, FaceObservation, derive_portrait_crop

    decision = derive_portrait_crop(
        (3840, 2160),
        [
            FaceObservation(
                0.0,
                primary=Box(1000, 400, 500, 600),
                secondary=Box(1800, 420, 470, 580),
            )
        ],
    )

    assert not decision.accepted
    assert decision.reason == "prominent_second_face"


def test_crop_rejects_insufficient_face_detection_coverage() -> None:
    from a2v_broad_dataset import Box, FaceObservation, derive_portrait_crop

    decision = derive_portrait_crop(
        (1920, 1080),
        [
            FaceObservation(0.0, primary=Box(700, 200, 300, 360)),
            FaceObservation(1.0, primary=None),
            FaceObservation(2.0, primary=None),
            FaceObservation(3.0, primary=None),
        ],
    )

    assert not decision.accepted
    assert decision.reason == "insufficient_face_coverage"
    assert decision.detection_coverage == 0.25


def test_map_proxy_crop_back_to_display_coordinates() -> None:
    from a2v_broad_dataset import Crop, map_crop_to_display

    mapped = map_crop_to_display(
        Crop(x=54, y=96, width=432, height=768),
        proxy_size=(540, 960),
        display_size=(2160, 3840),
    )

    assert mapped == Crop(x=216, y=384, width=1728, height=3072)
    assert abs(mapped.width / mapped.height - 9 / 16) < 0.001


def test_render_group_preserves_exact_a2v_contract() -> None:
    from a2v_broad_dataset import Crop, Window, render_group, validate_group

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1920x1080:rate=60",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
        )
        paths = render_group(
            source=source,
            window=Window(0.5, 0.5 + 89 / 24, word_count=8, speech_seconds=2.5),
            crop=Crop(656, 0, 608, 1080),
            destination=root / "group",
            basename="train_001",
            caption="A person speaks naturally to the camera in a bright room.",
        )

        audit = validate_group(paths)

        assert (audit.width, audit.height) == (544, 960)
        assert audit.frames == 89
        assert audit.fps == 24
        assert audit.audio_rate == 48_000
        assert audit.audio_channels == 1
        assert audit.start_matches_target_first_frame
        assert not audit.target_has_audio


def test_provider_mirror_inverts_only_visual_pixels() -> None:
    from a2v_broad_dataset import (
        Crop,
        Window,
        build_provider_mirror,
        render_group,
        write_training_archive,
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
        )
        canonical = root / "canonical"
        render_group(
            source=source,
            window=Window(0.5, 0.5 + 89 / 24, word_count=8, speech_seconds=2.5),
            crop=Crop(218, 0, 202, 360),
            destination=canonical,
            basename="train_001",
            caption="A person speaks naturally to the camera in a bright room.",
        )

        mirror = root / "mirror"
        audit = build_provider_mirror(canonical, mirror)

        assert audit.group_count == 1
        assert audit.audio_sha256_equal
        assert audit.caption_sha256_equal
        assert audit.visual_inverse_mean_absolute_error < 5.0

        archive = root / "training.zip"
        archive_audit = write_training_archive(mirror, archive)
        with zipfile.ZipFile(archive) as bundle:
            names = sorted(bundle.namelist())

        assert archive_audit.group_count == 1
        assert archive_audit.file_count == 4
        assert names == [
            "train_001.txt",
            "train_001_audio.wav",
            "train_001_end.mp4",
            "train_001_start.png",
        ]


def test_cli_dry_run_writes_source_disjoint_plan_without_media() -> None:
    from a2v_broad_dataset import main

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        approved = root / "approved.json"
        output = root / "output"
        source_root = root / "sources"
        source_root.mkdir()
        _write_approved_manifest(approved)

        assert (
            main(
                [
                    "--source-root",
                    str(source_root),
                    "--approved-manifest",
                    str(approved),
                    "--output-root",
                    str(output),
                    "--dry-run",
                ]
            )
            == 0
        )

        plan = json.loads((output / "dataset-plan.private.json").read_text())
        train_sources = {item["source_id"] for item in plan["training"]}
        holdout_sources = {item["source_id"] for item in plan["holdout"]}
        assert train_sources.isdisjoint(holdout_sources)
        assert len(holdout_sources) == 5
        assert not (output / "canonical-training").exists()
        assert not (output / "provider-mirror").exists()


def test_cli_rejects_projected_workspace_over_ceiling_before_render() -> None:
    from a2v_broad_dataset import main

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        approved = root / "approved.json"
        output = root / "output"
        source_root = root / "sources"
        source_root.mkdir()
        _write_approved_manifest(approved, source_count=10)

        with pytest.raises(ValueError, match="projected derived data"):
            main(
                [
                    "--source-root",
                    str(source_root),
                    "--approved-manifest",
                    str(approved),
                    "--output-root",
                    str(output),
                    "--projected-bytes-per-group",
                    "1024",
                    "--max-derived-bytes",
                    "1023",
                    "--render",
                ]
            )

        assert not (output / "canonical-training").exists()
