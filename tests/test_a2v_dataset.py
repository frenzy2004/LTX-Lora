from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import ltx_lora_pilot.a2v_dataset as a2v_dataset
from ltx_lora_pilot.a2v_dataset import A2VSpec, validate_a2v_directory


TEST_SPEC = A2VSpec(
    width=64,
    height=96,
    frames=9,
    fps=24,
    sample_rate=48_000,
    min_groups=1,
)


def _run_ffmpeg(arguments: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
    )


def _make_group(
    root: Path,
    *,
    group_id: str = "sample_001",
    caption: str = "A close-up speaker talks to camera.",
    silent: bool = False,
    target_has_audio: bool = False,
    target_codec: str = "libx264",
    target_format: str | None = None,
    target_size: str = "64x96",
    audio_codec: str = "pcm_s16le",
    audio_format: str | None = None,
    audio_channels: int = 1,
    audio_sample_rate: int = 48_000,
    target_start_offset: str = "0",
    first_frame_mismatch: bool = False,
    start_format: str | None = None,
    start_size: str | None = None,
    frames: int = 9,
    fps: int = 24,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{group_id}_end.mp4"
    duration = frames / fps
    target_arguments = [
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s={target_size}:r={fps}",
    ]
    if target_has_audio:
        target_arguments.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=880:sample_rate=48000",
            ]
        )
    target_arguments.extend(
        [
            "-map",
            "0:v:0",
            "-frames:v",
            str(frames),
            "-c:v",
            target_codec,
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if target_has_audio:
        target_arguments.extend(["-map", "1:a:0", "-c:a", "aac", "-shortest"])
    else:
        target_arguments.append("-an")
    if target_start_offset != "0":
        target_arguments.extend(["-output_ts_offset", target_start_offset])
    if target_format is not None:
        target_arguments.extend(["-f", target_format])
    target_arguments.append(str(target))
    _run_ffmpeg(target_arguments)

    start = root / f"{group_id}_start.png"
    if first_frame_mismatch:
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x96",
                "-frames:v",
                "1",
                str(start),
            ]
        )
    else:
        start_arguments = ["-i", str(target), "-frames:v", "1"]
        if start_size is not None:
            start_arguments.extend(["-vf", f"scale={start_size.replace('x', ':')}"])
        if start_format is not None:
            start_arguments.extend(["-c:v", "png", "-f", start_format])
        start_arguments.append(str(start))
        _run_ffmpeg(start_arguments)

    audio_source = (
        f"anullsrc=r={audio_sample_rate}:cl=mono"
        if silent
        else f"sine=frequency=440:sample_rate={audio_sample_rate}"
    )
    audio_arguments = [
            "-f",
            "lavfi",
            "-i",
            audio_source,
            "-t",
            str(duration),
            "-ac",
            str(audio_channels),
            "-c:a",
            audio_codec,
        ]
    if audio_format is not None:
        audio_arguments.extend(["-f", audio_format])
    audio_arguments.append(str(root / f"{group_id}_audio.wav"))
    _run_ffmpeg(audio_arguments)
    (root / f"{group_id}.txt").write_text(caption + "\n", encoding="utf-8")


def _validate(root: Path, *, trigger_phrase: str | None = None) -> dict:
    return validate_a2v_directory(
        root,
        spec=TEST_SPEC,
        trigger_phrase=trigger_phrase,
    )


def test_valid_a2v_group_returns_versioned_report_and_file_digests(
    tmp_path: Path,
) -> None:
    _make_group(tmp_path)

    report = _validate(tmp_path, trigger_phrase="chrx9_speech")

    expected_files = []
    for path in sorted(tmp_path.iterdir(), key=lambda item: item.name):
        content = path.read_bytes()
        expected_files.append(
            {
                "name": path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    assert report == {
        "schema_version": "a2v-structural-report-v1",
        "status": "valid",
        "spec": {
            "width": 64,
            "height": 96,
            "frames": 9,
            "fps": 24,
            "sample_rate": 48_000,
        },
        "groups": [{"group_id": "sample_001", "files": expected_files}],
    }


def test_a2v_caption_must_not_double_inject_trigger(tmp_path: Path) -> None:
    _make_group(tmp_path, caption="chrx9_speech speaking to camera")

    with pytest.raises(ValueError, match="already contains"):
        _validate(tmp_path, trigger_phrase="chrx9_speech")


def test_a2v_group_rejects_digital_silence(tmp_path: Path) -> None:
    _make_group(tmp_path, silent=True)

    with pytest.raises(ValueError, match="digital silence"):
        _validate(tmp_path)


def test_a2v_group_rejects_audio_stream_in_target(tmp_path: Path) -> None:
    _make_group(tmp_path, target_has_audio=True)

    with pytest.raises(ValueError, match="target must not contain audio"):
        _validate(tmp_path)


def test_a2v_group_rejects_wrong_video_codec(tmp_path: Path) -> None:
    _make_group(tmp_path, target_codec="mpeg4")

    with pytest.raises(ValueError, match="target video codec must be h264"):
        _validate(tmp_path)


def test_a2v_group_rejects_non_mp4_target_container(tmp_path: Path) -> None:
    _make_group(tmp_path, target_format="matroska")

    with pytest.raises(ValueError, match="target container must be MP4"):
        _validate(tmp_path)


@pytest.mark.parametrize("target_format", ["mov", "3gp"])
def test_a2v_group_rejects_other_isobmff_container_disguised_as_mp4(
    tmp_path: Path,
    target_format: str,
) -> None:
    _make_group(tmp_path, target_format=target_format)

    with pytest.raises(ValueError, match="target container must be MP4"):
        _validate(tmp_path)


def test_a2v_group_rejects_non_png_start_container(tmp_path: Path) -> None:
    _make_group(tmp_path, start_format="matroska")

    with pytest.raises(ValueError, match="start image container must be a standalone PNG"):
        _validate(tmp_path)


def test_a2v_group_rejects_non_pcm_wav(tmp_path: Path) -> None:
    _make_group(tmp_path, audio_codec="pcm_f32le")

    with pytest.raises(ValueError, match="PCM signed 16-bit"):
        _validate(tmp_path)


def test_a2v_group_rejects_non_wav_audio_container(tmp_path: Path) -> None:
    _make_group(tmp_path, audio_format="caf")

    with pytest.raises(ValueError, match="audio container must be WAV"):
        _validate(tmp_path)


def test_a2v_group_rejects_stereo_audio(tmp_path: Path) -> None:
    _make_group(tmp_path, audio_channels=2)

    with pytest.raises(ValueError, match="audio must be mono"):
        _validate(tmp_path)


def test_a2v_group_rejects_wrong_audio_sample_rate(tmp_path: Path) -> None:
    _make_group(tmp_path, audio_sample_rate=44_100)

    with pytest.raises(ValueError, match="audio sample rate is not 48000 Hz"):
        _validate(tmp_path)


@pytest.mark.parametrize(
    ("target_size", "start_size", "message"),
    [
        ("32x96", None, "target dimensions do not match 64x96"),
        ("64x96", "32x96", "start dimensions do not match 64x96"),
    ],
)
def test_a2v_group_rejects_wrong_dimensions(
    tmp_path: Path,
    target_size: str,
    start_size: str | None,
    message: str,
) -> None:
    _make_group(tmp_path, target_size=target_size, start_size=start_size)

    with pytest.raises(ValueError, match=message):
        _validate(tmp_path)


def test_a2v_group_rejects_nonzero_target_timestamp(tmp_path: Path) -> None:
    _make_group(tmp_path, target_start_offset="0.125")

    with pytest.raises(ValueError, match="target does not start at timestamp zero"):
        _validate(tmp_path)


def test_a2v_group_rejects_missing_target_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_group(tmp_path)
    original_ffprobe = a2v_dataset._ffprobe

    def ffprobe(path: Path, **kwargs: object) -> dict:
        payload = original_ffprobe(path, **kwargs)
        if path.name.endswith("_end.mp4"):
            for stream in payload["streams"]:
                if stream.get("codec_type") == "video":
                    stream.pop("start_time", None)
        return payload

    monkeypatch.setattr(a2v_dataset, "_ffprobe", ffprobe)

    with pytest.raises(ValueError, match="invalid start_time"):
        _validate(tmp_path)


def test_a2v_group_rejects_nonzero_audio_packet_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_group(tmp_path)
    original_ffprobe = a2v_dataset._ffprobe

    def ffprobe(path: Path, **kwargs: object) -> dict:
        payload = original_ffprobe(path, **kwargs)
        if path.name.endswith("_audio.wav"):
            for stream in payload["streams"]:
                if stream.get("codec_type") == "audio":
                    stream.pop("start_time", None)
                    stream["time_base"] = "1/48000"
            payload["packets"] = [{"codec_type": "audio", "pts": "1"}]
        return payload

    monkeypatch.setattr(a2v_dataset, "_ffprobe", ffprobe)

    with pytest.raises(ValueError, match="audio does not start at timestamp zero"):
        _validate(tmp_path)


def test_a2v_group_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_group(tmp_path)
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path.name == "sample_001.txt" or original_is_symlink(path)

    # Windows test sessions may lack the OS privilege to create a real symlink.
    # This replaces only the metadata query and still exercises the public gate.
    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(ValueError, match="symlink"):
        _validate(tmp_path)


def test_a2v_group_rejects_first_frame_mismatch(tmp_path: Path) -> None:
    _make_group(tmp_path, first_frame_mismatch=True)

    with pytest.raises(ValueError, match="decoded first target frame"):
        _validate(tmp_path)


def test_a2v_group_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    _make_group(tmp_path, frames=8)

    with pytest.raises(ValueError, match="8 frames, expected 9"):
        _validate(tmp_path)


def test_a2v_group_rejects_fps_mismatch(tmp_path: Path) -> None:
    _make_group(tmp_path, fps=25)

    with pytest.raises(ValueError, match="not 24 fps"):
        _validate(tmp_path)


def test_a2v_group_rejects_variable_frame_timestamps_at_nominal_24_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_group(tmp_path)
    original_ffprobe = a2v_dataset._ffprobe
    irregular_timestamps = [0, 1_000, 4_000, 5_000, 8_000, 9_000, 12_000, 13_000, 16_000]

    def ffprobe(path: Path, **kwargs: object) -> dict:
        payload = original_ffprobe(path, **kwargs)
        if path.name.endswith("_end.mp4"):
            for stream in payload["streams"]:
                if stream.get("codec_type") == "video":
                    stream["time_base"] = "1/48000"
            payload["frames"] = [
                {
                    "media_type": "video",
                    "best_effort_timestamp": str(timestamp),
                }
                for timestamp in irregular_timestamps
            ]
        return payload

    monkeypatch.setattr(a2v_dataset, "_ffprobe", ffprobe)

    with pytest.raises(ValueError, match="constant 24 fps"):
        _validate(tmp_path)


def test_a2v_directory_rejects_unexpected_file(tmp_path: Path) -> None:
    _make_group(tmp_path)
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected entries"):
        _validate(tmp_path)


def test_a2v_directory_rejects_unexpected_directory(tmp_path: Path) -> None:
    _make_group(tmp_path)
    (tmp_path / "nested").mkdir()

    with pytest.raises(ValueError, match="unexpected entries"):
        _validate(tmp_path)


def test_a2v_group_rejects_unsafe_group_id(tmp_path: Path) -> None:
    _make_group(tmp_path, group_id="Not Neutral")

    with pytest.raises(ValueError, match="unsafe group ID"):
        _validate(tmp_path)
