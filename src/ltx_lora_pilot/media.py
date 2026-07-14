from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass(frozen=True)
class MediaRecord:
    source_id: str
    path: str
    bytes: int
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _source_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


def probe(path: Path) -> MediaRecord:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
    numerator, denominator = video.get("avg_frame_rate", "0/1").split("/", 1)
    fps = float(numerator) / float(denominator or 1)
    return MediaRecord(
        source_id=_source_id(path),
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        duration=float(payload["format"]["duration"]),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        has_audio=any(stream.get("codec_type") == "audio" for stream in payload["streams"]),
    )


def inventory(source: Path) -> list[MediaRecord]:
    paths = sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    records = []
    for path in paths:
        try:
            records.append(probe(path))
        except (subprocess.CalledProcessError, StopIteration, KeyError, ValueError):
            continue
    return records


def write_inventory(records: list[MediaRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n", encoding="utf-8")
