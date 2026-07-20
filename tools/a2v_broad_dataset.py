from __future__ import annotations

import argparse
import json
import hashlib
import random
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    word_count: int
    speech_seconds: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class FaceObservation:
    time_seconds: float
    primary: Box | None
    secondary: Box | None = None


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int

    def contains(self, box: Box | None) -> bool:
        return bool(
            box is not None
            and box.x >= self.x
            and box.y >= self.y
            and box.right <= self.x + self.width
            and box.bottom <= self.y + self.height
        )


@dataclass(frozen=True)
class CropDecision:
    accepted: bool
    crop: Crop | None
    reason: str
    detection_coverage: float
    median_face_height_ratio: float | None


@dataclass(frozen=True)
class GroupPaths:
    target: Path
    start: Path
    audio: Path
    caption: Path


@dataclass(frozen=True)
class GroupAudit:
    width: int
    height: int
    frames: int
    fps: Fraction
    audio_rate: int
    audio_channels: int
    audio_samples: int
    start_matches_target_first_frame: bool
    target_has_audio: bool


@dataclass(frozen=True)
class MirrorAudit:
    group_count: int
    audio_sha256_equal: bool
    caption_sha256_equal: bool
    visual_inverse_mean_absolute_error: float


@dataclass(frozen=True)
class ArchiveAudit:
    group_count: int
    file_count: int
    size_bytes: int
    sha256: str
    member_names: tuple[str, ...]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _probe(path: Path, *, count_frames: bool = False) -> dict[str, object]:
    command = ["ffprobe", "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command.extend(["-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(_run(command).stdout.decode("utf-8"))


def _first_frame_rgb(path: Path) -> bytes:
    return _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    ).stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_for_basename(directory: Path, basename: str) -> GroupPaths:
    return GroupPaths(
        target=directory / f"{basename}_end.mp4",
        start=directory / f"{basename}_start.png",
        audio=directory / f"{basename}_audio.wav",
        caption=directory / f"{basename}.txt",
    )


def build_provider_mirror(canonical_dir: Path, mirror_dir: Path) -> MirrorAudit:
    """Pre-invert visual pixels while preserving audio and captions byte-for-byte."""

    canonical_dir = Path(canonical_dir)
    mirror_dir = Path(mirror_dir)
    if not canonical_dir.is_dir():
        raise FileNotFoundError(canonical_dir)
    targets = sorted(canonical_dir.glob("*_end.mp4"))
    if not targets:
        raise ValueError("canonical directory contains no A2V groups")
    if mirror_dir.exists() and any(mirror_dir.iterdir()):
        raise ValueError("provider mirror destination must be empty")
    mirror_dir.mkdir(parents=True, exist_ok=True)

    audio_equal = True
    caption_equal = True
    inverse_errors: list[float] = []
    for target in targets:
        basename = target.name.removesuffix("_end.mp4")
        canonical = _paths_for_basename(canonical_dir, basename)
        missing = [path.name for path in canonical.__dict__.values() if not path.is_file()]
        if missing:
            raise ValueError(f"incomplete canonical group {basename}: {missing}")
        validate_group(canonical)
        mirror = _paths_for_basename(mirror_dir, basename)
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(canonical.target),
                "-map",
                "0:v:0",
                "-vf",
                "format=rgb24,lutrgb=r=negval:g=negval:b=negval,format=yuv420p",
                "-frames:v",
                "89",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-brand",
                "isom",
                "-y",
                str(mirror.target),
            ]
        )
        shutil.copyfile(canonical.audio, mirror.audio)
        shutil.copyfile(canonical.caption, mirror.caption)
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(mirror.target),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-y",
                str(mirror.start),
            ]
        )
        validate_group(mirror)

        audio_equal = audio_equal and _sha256(canonical.audio) == _sha256(mirror.audio)
        caption_equal = caption_equal and _sha256(canonical.caption) == _sha256(
            mirror.caption
        )
        canonical_pixels = _first_frame_rgb(canonical.target)
        mirror_pixels = _first_frame_rgb(mirror.target)
        if len(canonical_pixels) != len(mirror_pixels) or not canonical_pixels:
            raise ValueError("canonical and mirror frames have different decoded sizes")
        inverse_errors.append(
            sum(
                abs((255 - canonical_value) - mirror_value)
                for canonical_value, mirror_value in zip(
                    canonical_pixels, mirror_pixels, strict=True
                )
            )
            / len(canonical_pixels)
        )

    return MirrorAudit(
        group_count=len(targets),
        audio_sha256_equal=audio_equal,
        caption_sha256_equal=caption_equal,
        visual_inverse_mean_absolute_error=sum(inverse_errors) / len(inverse_errors),
    )


def write_training_archive(groups_dir: Path, archive: Path) -> ArchiveAudit:
    """Write a deterministic ZIP containing exactly four files per valid group."""

    groups_dir = Path(groups_dir)
    archive = Path(archive)
    if not groups_dir.is_dir():
        raise FileNotFoundError(groups_dir)
    targets = sorted(groups_dir.glob("*_end.mp4"))
    if not targets:
        raise ValueError("groups directory contains no A2V targets")

    expected: dict[str, Path] = {}
    for target in targets:
        basename = target.name.removesuffix("_end.mp4")
        paths = _paths_for_basename(groups_dir, basename)
        for path in paths.__dict__.values():
            if not path.is_file():
                raise ValueError(f"incomplete A2V group {basename}: missing {path.name}")
            expected[path.name] = path
        validate_group(paths)

    actual = {
        path.name: path
        for path in groups_dir.iterdir()
        if path.is_file()
    }
    if set(actual) != set(expected):
        extras = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"unexpected archive membership; extras={extras}, missing={missing}")
    if any(path.is_dir() for path in groups_dir.iterdir()):
        raise ValueError("groups directory must not contain subdirectories")

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for name in sorted(expected):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                bundle.writestr(info, expected[name].read_bytes())
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()

    names = tuple(sorted(expected))
    return ArchiveAudit(
        group_count=len(targets),
        file_count=len(names),
        size_bytes=archive.stat().st_size,
        sha256=_sha256(archive),
        member_names=names,
    )


def render_group(
    *,
    source: Path,
    window: Window,
    crop: Crop,
    destination: Path,
    basename: str,
    caption: str,
    output_width: int = 544,
    output_height: int = 960,
    output_fps: int = 24,
    output_frames: int = 89,
    audio_rate: int = 48_000,
) -> GroupPaths:
    """Render one exact A2V group and fail if its media contract is invalid."""

    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not basename or Path(basename).name != basename:
        raise ValueError("basename must be a non-empty filename stem")
    if not caption.strip():
        raise ValueError("caption must not be empty")
    expected_duration = output_frames / output_fps
    if abs(window.duration - expected_duration) > 1e-6:
        raise ValueError("window duration must match the requested frame bucket")
    destination.mkdir(parents=True, exist_ok=True)
    paths = GroupPaths(
        target=destination / f"{basename}_end.mp4",
        start=destination / f"{basename}_start.png",
        audio=destination / f"{basename}_audio.wav",
        caption=destination / f"{basename}.txt",
    )

    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{window.start:.9f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            (
                f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
                f"scale={output_width}:{output_height}:flags=lanczos,"
                f"fps={output_fps},setpts=PTS-STARTPTS"
            ),
            "-frames:v",
            str(output_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-brand",
            "isom",
            "-y",
            str(paths.target),
        ]
    )

    audio_samples = round(expected_duration * audio_rate)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{window.start:.9f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            (
                f"aresample={audio_rate},aformat=sample_rates={audio_rate}:"
                "channel_layouts=mono,"
                f"apad=whole_len={audio_samples},atrim=end_sample={audio_samples},"
                "asetpts=N/SR/TB"
            ),
            "-ac",
            "1",
            "-ar",
            str(audio_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(paths.audio),
        ]
    )

    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(paths.target),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-y",
            str(paths.start),
        ]
    )
    paths.caption.write_text(caption.strip() + "\n", encoding="utf-8")
    validate_group(
        paths,
        expected_width=output_width,
        expected_height=output_height,
        expected_frames=output_frames,
        expected_fps=output_fps,
        expected_audio_rate=audio_rate,
    )
    return paths


def validate_group(
    paths: GroupPaths,
    *,
    expected_width: int = 544,
    expected_height: int = 960,
    expected_frames: int = 89,
    expected_fps: int = 24,
    expected_audio_rate: int = 48_000,
) -> GroupAudit:
    """Validate the exact frame, audio, start-frame, and stream contract."""

    target_probe = _probe(paths.target, count_frames=True)
    target_streams = target_probe.get("streams", [])
    if not isinstance(target_streams, list):
        raise ValueError("ffprobe returned invalid target streams")
    video_streams = [
        stream
        for stream in target_streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise ValueError("target must contain exactly one video stream")
    video = video_streams[0]
    target_has_audio = any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio"
        for stream in target_streams
    )
    width = int(video["width"])
    height = int(video["height"])
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    fps = Fraction(str(video.get("avg_frame_rate") or video.get("r_frame_rate")))

    audio_probe = _probe(paths.audio)
    audio_streams = audio_probe.get("streams", [])
    if not isinstance(audio_streams, list):
        raise ValueError("ffprobe returned invalid audio streams")
    audio_candidates = [
        stream
        for stream in audio_streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if len(audio_candidates) != 1:
        raise ValueError("audio file must contain exactly one audio stream")
    audio = audio_candidates[0]
    audio_rate = int(audio["sample_rate"])
    audio_channels = int(audio["channels"])
    audio_samples = int(audio.get("duration_ts") or 0)
    expected_audio_samples = round(expected_frames / expected_fps * expected_audio_rate)
    start_matches = _first_frame_rgb(paths.target) == _first_frame_rgb(paths.start)

    errors: list[str] = []
    if video.get("codec_name") != "h264":
        errors.append("target codec is not H.264")
    if (width, height) != (expected_width, expected_height):
        errors.append("target dimensions do not match the bucket")
    if frames != expected_frames:
        errors.append("target frame count does not match the bucket")
    if fps != expected_fps:
        errors.append("target frame rate does not match the bucket")
    if target_has_audio:
        errors.append("target video contains an audio stream")
    if audio.get("codec_name") != "pcm_s16le":
        errors.append("audio codec is not PCM signed 16-bit little-endian")
    if audio_rate != expected_audio_rate:
        errors.append("audio sample rate does not match the contract")
    if audio_channels != 1:
        errors.append("audio is not mono")
    if audio_samples != expected_audio_samples:
        errors.append("audio sample count does not match the frame bucket")
    if not start_matches:
        errors.append("start image does not match target frame zero")
    if not paths.caption.read_text(encoding="utf-8").strip():
        errors.append("caption is empty")
    if errors:
        raise ValueError("; ".join(errors))

    return GroupAudit(
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        audio_rate=audio_rate,
        audio_channels=audio_channels,
        audio_samples=audio_samples,
        start_matches_target_first_frame=start_matches,
        target_has_audio=target_has_audio,
    )


def _even_ceil(value: float) -> int:
    return int(ceil(value / 2.0) * 2)


def _even_clamped_origin(value: float, maximum: int) -> int:
    return max(0, min(maximum, int(round(value / 2.0) * 2)))


def _exact_even_aspect_dimensions(
    *,
    minimum_width: float,
    minimum_height: float,
    maximum_width: int,
    maximum_height: int,
    aspect: Fraction,
) -> tuple[int, int]:
    """Return even dimensions that exactly preserve a reduced aspect ratio."""

    if (
        minimum_width <= 0
        or minimum_height <= 0
        or maximum_width < 2
        or maximum_height < 2
        or aspect <= 0
    ):
        raise ValueError("aspect dimensions must be positive")
    width_unit = aspect.numerator
    height_unit = aspect.denominator
    unit_step = 2 if width_unit % 2 or height_unit % 2 else 1
    required_units = ceil(
        max(minimum_width / width_unit, minimum_height / height_unit)
        / unit_step
    ) * unit_step
    maximum_units = min(
        maximum_width // width_unit,
        maximum_height // height_unit,
    )
    maximum_units -= maximum_units % unit_step
    if maximum_units < unit_step:
        raise ValueError("frame is too small for the requested even aspect ratio")
    units = min(required_units, maximum_units)
    return width_unit * units, height_unit * units


def map_crop_to_display(
    crop: Crop,
    *,
    proxy_size: tuple[int, int],
    display_size: tuple[int, int],
) -> Crop:
    """Map a crop derived on a downscaled proxy back to display coordinates."""

    proxy_width, proxy_height = proxy_size
    display_width, display_height = display_size
    if min(proxy_width, proxy_height, display_width, display_height) < 2:
        raise ValueError("proxy and display dimensions must be positive")
    if (
        crop.x < 0
        or crop.y < 0
        or crop.width < 2
        or crop.height < 2
        or crop.x + crop.width > proxy_width
        or crop.y + crop.height > proxy_height
    ):
        raise ValueError("proxy crop lies outside the proxy frame")
    proxy_aspect = proxy_width / proxy_height
    display_aspect = display_width / display_height
    if abs(proxy_aspect - display_aspect) / display_aspect > 0.01:
        raise ValueError("proxy and display aspect ratios do not match")

    scale_x = display_width / proxy_width
    scale_y = display_height / proxy_height
    width, height = _exact_even_aspect_dimensions(
        minimum_width=crop.width * scale_x,
        minimum_height=crop.height * scale_y,
        maximum_width=display_width,
        maximum_height=display_height,
        aspect=Fraction(crop.width, crop.height),
    )
    center_x = (crop.x + crop.width / 2) * scale_x
    center_y = (crop.y + crop.height / 2) * scale_y
    x = _even_clamped_origin(center_x - width / 2, display_width - width)
    y = _even_clamped_origin(center_y - height / 2, display_height - height)
    return Crop(x=x, y=y, width=width, height=height)


def derive_portrait_crop(
    frame_size: tuple[int, int],
    observations: Sequence[FaceObservation],
    *,
    output_aspect: float = 544 / 960,
    target_face_height_ratio: float = 0.28,
    minimum_detection_coverage: float = 0.75,
    prominent_second_face_ratio: float = 0.35,
) -> CropDecision:
    """Derive one fixed portrait crop from a face's temporal envelope."""

    frame_width, frame_height = frame_size
    if frame_width < 2 or frame_height < 2 or not observations:
        raise ValueError("a valid frame and at least one observation are required")
    primary_boxes = [item.primary for item in observations if item.primary is not None]
    coverage = len(primary_boxes) / len(observations)
    if coverage < minimum_detection_coverage:
        return CropDecision(False, None, "insufficient_face_coverage", coverage, None)

    if any(
        item.primary is not None
        and item.secondary is not None
        and item.secondary.area / item.primary.area >= prominent_second_face_ratio
        for item in observations
    ):
        return CropDecision(False, None, "prominent_second_face", coverage, None)

    boxes = [box for box in primary_boxes if box is not None]
    median_width = median(box.width for box in boxes)
    median_height = median(box.height for box in boxes)
    envelope_left = min(box.x for box in boxes)
    envelope_top = min(box.y for box in boxes)
    envelope_right = max(box.right for box in boxes)
    envelope_bottom = max(box.bottom for box in boxes)

    required_width = envelope_right - envelope_left + median_width
    required_height = envelope_bottom - envelope_top + median_height
    crop_height = max(
        median_height / target_face_height_ratio,
        required_height,
        required_width / output_aspect,
    )
    crop_width = crop_height * output_aspect
    if crop_height > frame_height:
        crop_height = float(frame_height)
        crop_width = crop_height * output_aspect
    if crop_width > frame_width:
        crop_width = float(frame_width)
        crop_height = crop_width / output_aspect

    width, height = _exact_even_aspect_dimensions(
        minimum_width=crop_width,
        minimum_height=crop_height,
        maximum_width=frame_width,
        maximum_height=frame_height,
        aspect=Fraction(str(output_aspect)).limit_denominator(1_000),
    )

    face_center_x = median(box.center_x for box in boxes)
    face_center_y = median(box.center_y for box in boxes)
    desired_x = face_center_x - width / 2
    desired_y = face_center_y - height * 0.33
    desired_x = min(desired_x, envelope_left)
    desired_x = max(desired_x, envelope_right - width)
    desired_y = min(desired_y, envelope_top)
    desired_y = max(desired_y, envelope_bottom - height)
    x = _even_clamped_origin(desired_x, frame_width - width)
    y = _even_clamped_origin(desired_y, frame_height - height)
    crop = Crop(x=x, y=y, width=width, height=height)
    if not all(crop.contains(box) for box in boxes):
        return CropDecision(False, None, "face_envelope_outside_crop", coverage, None)
    return CropDecision(
        True,
        crop,
        "accepted",
        coverage,
        median_height / height,
    )


def split_sources(
    source_ids: Sequence[str],
    *,
    holdout_fraction: float = 0.10,
    min_holdout: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return deterministic, source-disjoint training and holdout identifiers."""

    ordered = sorted(set(source_ids))
    if len(ordered) < 2:
        raise ValueError("at least two distinct sources are required")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    if min_holdout < 1:
        raise ValueError("min_holdout must be positive")

    holdout_count = min(
        len(ordered) - 1,
        max(min_holdout, round(len(ordered) * holdout_fraction)),
    )
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    holdout = sorted(shuffled[:holdout_count])
    holdout_set = set(holdout)
    train = [source_id for source_id in ordered if source_id not in holdout_set]
    return train, holdout


def select_speech_windows(
    words: Sequence[Word],
    *,
    source_duration: float,
    clip_seconds: float = 89 / 24,
    max_windows: int = 2,
) -> list[Window]:
    """Select exact-duration, non-overlapping windows with dense recognized speech."""

    if source_duration < clip_seconds:
        return []
    if clip_seconds <= 0 or max_windows < 1:
        raise ValueError("clip_seconds and max_windows must be positive")

    valid_words = sorted(
        (
            word
            for word in words
            if 0.0 <= word.start < word.end <= source_duration and word.text.strip()
        ),
        key=lambda word: (word.start, word.end, word.text),
    )
    if not valid_words:
        return []

    latest_start = source_duration - clip_seconds
    starts = {0.0, latest_start}
    for word in valid_words:
        starts.add(min(latest_start, max(0.0, word.start)))
        starts.add(min(latest_start, max(0.0, word.end - clip_seconds)))

    candidates: list[Window] = []
    for start in starts:
        end = start + clip_seconds
        contained = [
            word for word in valid_words if word.start >= start and word.end <= end
        ]
        if not contained:
            continue
        candidates.append(
            Window(
                start=start,
                end=end,
                word_count=len(contained),
                speech_seconds=sum(word.end - word.start for word in contained),
            )
        )

    candidates.sort(
        key=lambda window: (
            -window.word_count,
            -window.speech_seconds,
            window.start,
        )
    )
    selected: list[Window] = []
    for candidate in candidates:
        if any(
            candidate.start < existing.end and candidate.end > existing.start
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == max_windows:
            break
    return sorted(selected, key=lambda window: window.start)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_relative_source(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("source_relative_path must stay inside source root")
    candidate = (root / relative_path).resolve(strict=True)
    candidate.relative_to(root)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _normalise_approved_windows(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), list):
        raise ValueError("approved manifest must contain a windows list")
    accepted: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    expected_duration = 89 / 24
    for raw in payload["windows"]:
        if not isinstance(raw, dict):
            raise ValueError("every approved window must be an object")
        if raw.get("visual_status") != "accepted":
            raise ValueError("every planned window requires explicit visual acceptance")
        source_id = str(raw.get("source_id", "")).strip()
        relative = str(raw.get("source_relative_path", "")).strip()
        caption = str(raw.get("caption", "")).strip()
        window_index = int(raw.get("window_index", 0))
        start = float(raw.get("start", -1))
        end = float(raw.get("end", -1))
        crop_payload = raw.get("crop")
        if not source_id or not relative or not caption or window_index < 1 or start < 0:
            raise ValueError("approved window metadata is incomplete")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("source_relative_path must be safe and relative")
        if abs((end - start) - expected_duration) > 1e-6:
            raise ValueError("approved window does not match the 89-frame duration")
        if not isinstance(crop_payload, dict):
            raise ValueError("approved window is missing its crop")
        crop = Crop(
            x=int(crop_payload.get("x", -1)),
            y=int(crop_payload.get("y", -1)),
            width=int(crop_payload.get("width", 0)),
            height=int(crop_payload.get("height", 0)),
        )
        if (
            crop.x < 0
            or crop.y < 0
            or crop.width < 2
            or crop.height < 2
            or any(value % 2 for value in (crop.x, crop.y, crop.width, crop.height))
        ):
            raise ValueError("crop coordinates and dimensions must be nonnegative/even")
        if crop.width * 30 != crop.height * 17:
            raise ValueError("crop aspect must exactly match the 544x960 bucket")
        key = (source_id, window_index)
        if key in seen:
            raise ValueError(f"duplicate approved window: {source_id} w{window_index}")
        seen.add(key)
        accepted.append(
            {
                "source_id": source_id,
                "source_relative_path": relative_path.as_posix(),
                "window_index": window_index,
                "start": start,
                "end": end,
                "word_count": int(raw.get("word_count", 0)),
                "speech_seconds": float(raw.get("speech_seconds", 0.0)),
                "crop": {
                    "x": crop.x,
                    "y": crop.y,
                    "width": crop.width,
                    "height": crop.height,
                },
                "caption": caption,
                "visual_status": "accepted",
            }
        )
    if len({str(item["source_id"]) for item in accepted}) < 2:
        raise ValueError("at least two visually accepted source files are required")
    return sorted(
        accepted,
        key=lambda item: (str(item["source_id"]), int(item["window_index"])),
    )


def build_dataset_plan(
    approved_windows: Sequence[dict[str, object]],
    *,
    projected_bytes_per_group: int,
    max_derived_bytes: int,
    holdout_fraction: float = 0.10,
    min_holdout: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Split visually approved windows by source and enforce the size ceiling."""

    if projected_bytes_per_group < 1 or max_derived_bytes < 1:
        raise ValueError("workspace size limits must be positive")
    source_ids = sorted({str(item["source_id"]) for item in approved_windows})
    train_sources, holdout_sources = split_sources(
        source_ids,
        holdout_fraction=holdout_fraction,
        min_holdout=min_holdout,
        seed=seed,
    )
    train_set = set(train_sources)
    holdout_set = set(holdout_sources)
    training: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for item in approved_windows:
        source_id = str(item["source_id"])
        target = training if source_id in train_set else holdout
        if source_id not in train_set and source_id not in holdout_set:
            raise ValueError(f"source missing from split: {source_id}")
        planned = dict(item)
        prefix = "train" if target is training else "holdout"
        planned["basename"] = f"{prefix}_{len(target) + 1:03d}"
        target.append(planned)
    if {str(item["source_id"]) for item in training} & {
        str(item["source_id"]) for item in holdout
    }:
        raise ValueError("training and holdout sources overlap")
    projected_size = len(approved_windows) * projected_bytes_per_group
    if projected_size > max_derived_bytes:
        raise ValueError(
            "projected derived data exceeds the configured workspace ceiling: "
            f"{projected_size} > {max_derived_bytes}"
        )
    return {
        "schema_version": 1,
        "bucket": {"width": 544, "height": 960, "frames": 89, "fps": 24},
        "split": {
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "min_holdout": min_holdout,
        },
        "source_count": len(source_ids),
        "training_source_count": len(train_sources),
        "holdout_source_count": len(holdout_sources),
        "training_group_count": len(training),
        "holdout_group_count": len(holdout),
        "projected_bytes_per_group": projected_bytes_per_group,
        "projected_derived_bytes": projected_size,
        "max_derived_bytes": max_derived_bytes,
        "training": training,
        "holdout": holdout,
    }


def _render_planned_group(
    item: dict[str, object],
    *,
    source_root: Path,
    destination: Path,
) -> GroupAudit:
    source = _safe_relative_source(source_root, str(item["source_relative_path"]))
    crop_payload = item["crop"]
    if not isinstance(crop_payload, dict):
        raise ValueError("planned crop is invalid")
    paths = _paths_for_basename(destination, str(item["basename"]))
    membership = [path.is_file() for path in paths.__dict__.values()]
    if all(membership):
        return validate_group(paths)
    if any(membership):
        raise ValueError(f"partial existing group: {item['basename']}")
    return validate_group(
        render_group(
            source=source,
            window=Window(
                start=float(item["start"]),
                end=float(item["end"]),
                word_count=int(item["word_count"]),
                speech_seconds=float(item["speech_seconds"]),
            ),
            crop=Crop(
                x=int(crop_payload["x"]),
                y=int(crop_payload["y"]),
                width=int(crop_payload["width"]),
                height=int(crop_payload["height"]),
            ),
            destination=destination,
            basename=str(item["basename"]),
            caption=str(item["caption"]),
        )
    )


def render_dataset_plan(
    plan: dict[str, object],
    *,
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Render canonical groups and the provider-only mirror under a hard ceiling."""

    source_root = source_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    max_bytes = int(plan["max_derived_bytes"])
    training_dir = output_root / "canonical-training"
    holdout_dir = output_root / "canonical-holdout"
    audits: list[dict[str, object]] = []
    for split_name, destination in (
        ("training", training_dir),
        ("holdout", holdout_dir),
    ):
        items = plan.get(split_name)
        if not isinstance(items, list):
            raise ValueError(f"plan is missing {split_name} items")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("planned group must be an object")
            audit = _render_planned_group(
                item,
                source_root=source_root,
                destination=destination,
            )
            audits.append(
                {
                    "basename": item["basename"],
                    "split": split_name,
                    "width": audit.width,
                    "height": audit.height,
                    "frames": audit.frames,
                    "fps": str(audit.fps),
                    "audio_rate": audit.audio_rate,
                    "audio_channels": audit.audio_channels,
                    "audio_samples": audit.audio_samples,
                    "start_matches_target_first_frame": audit.start_matches_target_first_frame,
                    "target_has_audio": audit.target_has_audio,
                }
            )
            current_size = _directory_size(output_root)
            if current_size > max_bytes:
                raise ValueError(
                    "rendered derived data exceeded the workspace ceiling: "
                    f"{current_size} > {max_bytes}"
                )

    mirror_dir = output_root / "provider-mirror"
    if mirror_dir.exists() and any(mirror_dir.iterdir()):
        raise ValueError("provider mirror already exists and is not empty")
    mirror = build_provider_mirror(training_dir, mirror_dir)
    archive = write_training_archive(
        mirror_dir,
        output_root / "provider-training.zip",
    )
    actual_size = _directory_size(output_root)
    if actual_size > max_bytes:
        raise ValueError(
            "completed derived data exceeded the workspace ceiling: "
            f"{actual_size} > {max_bytes}"
        )
    return {
        "schema_version": 1,
        "group_audits": audits,
        "mirror": {
            "group_count": mirror.group_count,
            "audio_sha256_equal": mirror.audio_sha256_equal,
            "caption_sha256_equal": mirror.caption_sha256_equal,
            "visual_inverse_mean_absolute_error": mirror.visual_inverse_mean_absolute_error,
        },
        "archive": {
            "group_count": archive.group_count,
            "file_count": archive.file_count,
            "size_bytes": archive.size_bytes,
            "sha256": archive.sha256,
        },
        "actual_derived_bytes": actual_size,
        "max_derived_bytes": max_bytes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-isolated broad A2V training archive."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--approved-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--projected-bytes-per-group",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-derived-bytes",
        type=int,
        default=8 * 1024 * 1024 * 1024,
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--min-holdout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)

    approved_payload = json.loads(
        args.approved_manifest.resolve(strict=True).read_text(encoding="utf-8")
    )
    approved = _normalise_approved_windows(approved_payload)
    plan = build_dataset_plan(
        approved,
        projected_bytes_per_group=args.projected_bytes_per_group,
        max_derived_bytes=args.max_derived_bytes,
        holdout_fraction=args.holdout_fraction,
        min_holdout=args.min_holdout,
        seed=args.seed,
    )
    output_root = args.output_root.resolve()
    _write_json_atomic(output_root / "dataset-plan.private.json", plan)
    if args.dry_run:
        return 0
    build = render_dataset_plan(
        plan,
        source_root=args.source_root,
        output_root=output_root,
    )
    _write_json_atomic(output_root / "dataset-build.private.json", build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
