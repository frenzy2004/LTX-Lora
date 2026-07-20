from __future__ import annotations

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

    scale = display_height / proxy_height
    width = min(display_width, _even_ceil(crop.width * scale))
    height = min(display_height, _even_ceil(crop.height * scale))
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    x = _even_clamped_origin(crop.x * scale, display_width - width)
    y = _even_clamped_origin(crop.y * scale, display_height - height)
    return Crop(x=x, y=y, width=width, height=height)


def derive_portrait_crop(
    frame_size: tuple[int, int],
    observations: Sequence[FaceObservation],
    *,
    output_aspect: float = 9 / 16,
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

    width = min(frame_width, _even_ceil(crop_width))
    height = min(frame_height, _even_ceil(crop_height))
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1

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
