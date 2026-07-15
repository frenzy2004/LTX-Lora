from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from .artifacts import safe_relative_name, sha256_file


STRUCTURAL_REPORT_SCHEMA = "a2v-structural-report-v1"
GROUP_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}", re.ASCII)


@dataclass(frozen=True)
class A2VSpec:
    width: int = 544
    height: int = 960
    frames: int = 89
    fps: int = 24
    sample_rate: int = 48_000
    min_groups: int = 10


def _ffprobe(
    path: Path,
    *,
    count_frames: bool = False,
    show_frames: bool = False,
    show_packets: bool = False,
) -> dict:
    command = ["ffprobe", "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    if show_frames:
        command.append("-show_frames")
    if show_packets:
        command.append("-show_packets")
    show_entries = (
        "stream=codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,"
        "r_frame_rate,nb_read_frames,start_time,duration,sample_rate,"
        "channels,sample_fmt,time_base:format=format_name"
    )
    if show_frames:
        show_entries += ":frame=media_type,best_effort_timestamp"
    if show_packets:
        show_entries += ":packet=codec_type,pts"
    command.extend(
        [
            "-show_entries",
            show_entries,
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to validate A2V media") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"group media is invalid or undecodable: {path.name}") from exc
    if type(payload) is not dict or type(payload.get("streams")) is not list:
        raise ValueError(f"group media has an invalid probe result: {path.name}")
    return payload


def _decode_with_ffmpeg(path: Path, *arguments: str, purpose: str) -> bytes:
    command = ["ffmpeg", "-v", "error", "-i", str(path), *arguments, "-"]
    try:
        return subprocess.run(command, check=True, capture_output=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg is required to {purpose}") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"could not {purpose}: {path.name}") from exc


def _first_frame_sha256(path: Path, *, width: int, height: int) -> str:
    pixels = _decode_with_ffmpeg(
        path,
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        "format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        purpose="decode the first RGB frame",
    )
    expected_bytes = width * height * 3
    if len(pixels) != expected_bytes:
        raise ValueError(f"expected exactly one decoded first RGB frame: {path.name}")
    return hashlib.sha256(pixels).hexdigest()


def _reject_digital_silence(path: Path) -> None:
    samples = _decode_with_ffmpeg(
        path,
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        purpose="decode signed 16-bit audio samples",
    )
    if not samples or len(samples) % 2:
        raise ValueError(f"audio has no complete signed 16-bit samples: {path.name}")
    if not any(samples):
        raise ValueError(f"group audio is digital silence: {path.name}")


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _one_stream(payload: dict, *, codec_type: str, label: str) -> dict:
    streams = payload["streams"]
    matching = [stream for stream in streams if stream.get("codec_type") == codec_type]
    if len(streams) != 1 or len(matching) != 1:
        raise ValueError(f"{label} must contain exactly one {codec_type} stream")
    return matching[0]


def _format_names(payload: dict, *, label: str) -> set[str]:
    format_value = payload.get("format")
    if type(format_value) is not dict or type(format_value.get("format_name")) is not str:
        raise ValueError(f"{label} has an invalid container format")
    return set(format_value["format_name"].split(","))


def _decimal_field(stream: dict, field: str, *, label: str, default: str | None = None) -> Decimal:
    raw = stream.get(field, default)
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{label} has an invalid {field}") from exc


def _fraction_field(stream: dict, field: str, *, label: str) -> Fraction:
    try:
        return Fraction(stream[field])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} has an invalid {field}") from exc


def _integer_field(stream: dict, field: str, *, label: str) -> int:
    try:
        return int(stream[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid {field}") from exc


def _validate_constant_frame_timestamps(
    payload: dict,
    video: dict,
    *,
    group_id: str,
    spec: A2VSpec,
) -> None:
    frames = payload.get("frames")
    if type(frames) is not list:
        raise ValueError(f"group {group_id} target has no decoded frame timestamps")
    video_frames = [frame for frame in frames if frame.get("media_type") == "video"]
    if len(video_frames) != spec.frames:
        raise ValueError(f"group {group_id} target has incomplete decoded frame timestamps")
    try:
        time_base = Fraction(video["time_base"])
        timestamps = [int(frame["best_effort_timestamp"]) for frame in video_frames]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"group {group_id} target has invalid decoded frame timestamps") from exc
    if time_base <= 0:
        raise ValueError(f"group {group_id} target has invalid decoded frame timestamps")
    expected_delta = Fraction(1, spec.fps)
    if any(
        Fraction(right - left) * time_base != expected_delta
        for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(f"group {group_id} target is not constant {spec.fps} fps")


def _validate_zero_audio_timestamp(
    payload: dict,
    audio_stream: dict,
    *,
    group_id: str,
) -> None:
    if "start_time" in audio_stream and _decimal_field(
        audio_stream,
        "start_time",
        label=f"group {group_id} audio",
    ) != 0:
        raise ValueError(f"group {group_id} audio does not start at timestamp zero")
    packets = payload.get("packets")
    if type(packets) is not list:
        raise ValueError(f"group {group_id} audio has no decoded packet timestamps")
    audio_packets = [packet for packet in packets if packet.get("codec_type") == "audio"]
    if not audio_packets:
        raise ValueError(f"group {group_id} audio has no decoded packet timestamps")
    try:
        time_base = Fraction(audio_stream["time_base"])
        first_pts = int(audio_packets[0]["pts"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"group {group_id} audio has an invalid packet timestamp") from exc
    if time_base <= 0:
        raise ValueError(f"group {group_id} audio has an invalid packet timestamp")
    if Fraction(first_pts) * time_base != 0:
        raise ValueError(f"group {group_id} audio does not start at timestamp zero")


def _validate_spec(spec: A2VSpec) -> None:
    values = (
        spec.width,
        spec.height,
        spec.frames,
        spec.fps,
        spec.sample_rate,
        spec.min_groups,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("A2V specification values must be positive integers")


def validate_a2v_directory(
    root: Path,
    *,
    spec: A2VSpec = A2VSpec(),
    trigger_phrase: str | None = None,
) -> dict:
    _validate_spec(spec)
    if _is_symlink_or_junction(root):
        raise ValueError("A2V archive root must not be a symlink")
    if not root.is_dir():
        raise NotADirectoryError(root)

    entries = sorted(root.iterdir(), key=lambda path: path.name)
    for path in entries:
        if _is_symlink_or_junction(path):
            raise ValueError("A2V archive entries must not be symlinks")

    targets = [path for path in entries if path.name.endswith("_end.mp4")]
    if len(targets) < spec.min_groups:
        raise ValueError(f"need at least {spec.min_groups} A2V groups, found {len(targets)}")

    allowed: set[Path] = set()
    groups: list[dict] = []
    for target in targets:
        group_id = target.name[: -len("_end.mp4")]
        if GROUP_ID_PATTERN.fullmatch(group_id) is None:
            raise ValueError("unsafe group ID in A2V archive")

        start = root / f"{group_id}_start.png"
        audio = root / f"{group_id}_audio.wav"
        caption = root / f"{group_id}.txt"
        required = (start, audio, target, caption)
        missing = [path.name for path in required if not _is_regular_file(path)]
        if missing:
            raise ValueError(f"group {group_id} is missing regular files: {missing}")
        for path in required:
            safe_relative_name(path.name)
        allowed.update(required)

        try:
            text = caption.read_text(encoding="utf-8").strip()
        except UnicodeError as exc:
            raise ValueError(f"group {group_id} caption is not valid UTF-8") from exc
        if not text:
            raise ValueError(f"group {group_id} has an empty caption")
        if trigger_phrase and trigger_phrase in text:
            raise ValueError(f"group {group_id} caption already contains the managed trigger phrase")

        target_probe = _ffprobe(target, count_frames=True, show_frames=True)
        target_streams = target_probe["streams"]
        if any(stream.get("codec_type") == "audio" for stream in target_streams):
            raise ValueError(f"group {group_id} target must not contain audio")
        image_probe = _ffprobe(start)
        audio_probe = _ffprobe(audio, show_packets=True)
        video = _one_stream(target_probe, codec_type="video", label=f"group {group_id} target")
        image = _one_stream(
            image_probe,
            codec_type="video",
            label=f"group {group_id} start image",
        )
        audio_stream = _one_stream(
            audio_probe,
            codec_type="audio",
            label=f"group {group_id} audio",
        )

        if "mp4" not in _format_names(target_probe, label=f"group {group_id} target"):
            raise ValueError(f"group {group_id} target container must be MP4")
        if "png_pipe" not in _format_names(
            image_probe,
            label=f"group {group_id} start image",
        ):
            raise ValueError(
                f"group {group_id} start image container must be a standalone PNG"
            )
        if "wav" not in _format_names(audio_probe, label=f"group {group_id} audio"):
            raise ValueError(f"group {group_id} audio container must be WAV")
        if video.get("codec_name") != "h264":
            raise ValueError(f"group {group_id} target video codec must be h264")
        if image.get("codec_name") != "png" or image.get("pix_fmt") != "rgb24":
            raise ValueError(f"group {group_id} start image must be an RGB PNG")
        if audio_stream.get("codec_name") != "pcm_s16le" or audio_stream.get("sample_fmt") != "s16":
            raise ValueError(f"group {group_id} audio must be PCM signed 16-bit")
        if _integer_field(audio_stream, "channels", label=f"group {group_id} audio") != 1:
            raise ValueError(f"group {group_id} audio must be mono")

        if (
            _integer_field(video, "width", label=f"group {group_id} target"),
            _integer_field(video, "height", label=f"group {group_id} target"),
        ) != (spec.width, spec.height):
            raise ValueError(f"group {group_id} target dimensions do not match {spec.width}x{spec.height}")
        if (
            _integer_field(image, "width", label=f"group {group_id} start image"),
            _integer_field(image, "height", label=f"group {group_id} start image"),
        ) != (spec.width, spec.height):
            raise ValueError(f"group {group_id} start dimensions do not match {spec.width}x{spec.height}")
        frame_count = _integer_field(video, "nb_read_frames", label=f"group {group_id} target")
        if frame_count != spec.frames:
            raise ValueError(f"group {group_id} has {frame_count} frames, expected {spec.frames}")
        expected_fps = Fraction(spec.fps, 1)
        if (
            _fraction_field(video, "avg_frame_rate", label=f"group {group_id} target") != expected_fps
            or _fraction_field(video, "r_frame_rate", label=f"group {group_id} target") != expected_fps
        ):
            raise ValueError(f"group {group_id} target frame rate is not {spec.fps} fps")
        _validate_constant_frame_timestamps(
            target_probe,
            video,
            group_id=group_id,
            spec=spec,
        )
        if _integer_field(audio_stream, "sample_rate", label=f"group {group_id} audio") != spec.sample_rate:
            raise ValueError(f"group {group_id} audio sample rate is not {spec.sample_rate} Hz")
        if _decimal_field(video, "start_time", label=f"group {group_id} target") != 0:
            raise ValueError(f"group {group_id} target does not start at timestamp zero")
        _validate_zero_audio_timestamp(audio_probe, audio_stream, group_id=group_id)

        video_duration = _decimal_field(video, "duration", label=f"group {group_id} target")
        audio_duration = _decimal_field(audio_stream, "duration", label=f"group {group_id} audio")
        tolerance = Decimal(1) / Decimal(spec.fps)
        if abs(video_duration - audio_duration) > tolerance:
            raise ValueError(
                f"group {group_id} A/V duration delta exceeds one frame: "
                f"video={video_duration}, audio={audio_duration}"
            )
        if _first_frame_sha256(start, width=spec.width, height=spec.height) != _first_frame_sha256(
            target,
            width=spec.width,
            height=spec.height,
        ):
            raise ValueError(f"group {group_id} start image is not the decoded first target frame")
        _reject_digital_silence(audio)

        groups.append(
            {
                "group_id": group_id,
                "files": [asdict(sha256_file(path)) for path in sorted(required, key=lambda item: item.name)],
            }
        )

    unexpected = [path.name for path in entries if path not in allowed]
    if unexpected:
        raise ValueError(f"unexpected entries in A2V archive root: {unexpected}")

    return {
        "schema_version": STRUCTURAL_REPORT_SCHEMA,
        "status": "valid",
        "spec": {
            "width": spec.width,
            "height": spec.height,
            "frames": spec.frames,
            "fps": spec.fps,
            "sample_rate": spec.sample_rate,
        },
        "groups": groups,
    }
