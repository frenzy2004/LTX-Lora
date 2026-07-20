from __future__ import annotations

import sys
import subprocess
import tempfile
import zipfile
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "tools"))


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
    assert abs(decision.crop.width / decision.crop.height - 9 / 16) < 0.002
    assert decision.crop.x > (3840 - decision.crop.width) / 2
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
