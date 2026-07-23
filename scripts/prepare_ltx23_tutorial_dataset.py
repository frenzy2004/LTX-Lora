from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ltx_lora_pilot.ltx23_v2 import TRIGGER


BLOCKED_NAME_TOKENS = {"realname", "surname"}


def _clip_filter() -> str:
    return (
        "fps=24,"
        "crop='if(gt(iw/ih,9/16),ih*9/16,iw)':"
        "'if(gt(iw/ih,9/16),ih,iw*16/9)':"
        "'(iw-ow)/2':'(ih-oh)/2',"
        "scale=720:1280,"
        "setsar=1,"
        "format=yuv420p"
    )


def _clean_text(text: str) -> str:
    cleaned = text.replace('"', "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_caption(text: str) -> str:
    cleaned = _clean_text(text)
    words = []
    for word in cleaned.split(" "):
        normalized = re.sub(r"[^a-z]", "", word.lower())
        if normalized in BLOCKED_NAME_TOKENS:
            continue
        words.append(word)
    cleaned = " ".join(words).strip(" ,")
    if not cleaned:
        cleaned = "talks about a small everyday idea."
    cleaned = cleaned[0].upper() + cleaned[1:]
    return f'{TRIGGER} says, "{cleaned}"'


def _segment_duration(segment: dict[str, Any]) -> float:
    return float(segment["end"]) - float(segment["start"])


def _make_window(segments: list[dict[str, Any]]) -> dict[str, Any]:
    start = float(segments[0]["start"])
    end = float(segments[-1]["end"])
    text = " ".join(_clean_text(str(segment.get("text", ""))) for segment in segments).strip()
    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "text": text,
        "caption": sanitize_caption(text),
    }


def choose_clip_windows(
    segments: list[dict[str, Any]],
    *,
    target_count: int,
    min_seconds: float = 2.0,
    max_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    ordered = sorted(segments, key=lambda segment: float(segment["start"]))
    seen = set()

    for index, segment in enumerate(ordered):
        text = _clean_text(str(segment.get("text", "")))
        if _segment_duration(segment) >= min_seconds or text.endswith((".", "?", "!")):
            continue
        group = [segment]
        end = float(segment["end"])
        for neighbor in ordered[index + 1 :]:
            if float(neighbor["start"]) - end > 1.0:
                break
            next_end = float(neighbor["end"])
            if next_end - float(group[0]["start"]) > max_seconds:
                break
            group.append(neighbor)
            end = next_end
            duration = end - float(group[0]["start"])
            if duration >= min_seconds:
                window = _make_window(group)
                key = (round(float(window["start"]), 3), round(float(window["end"]), 3))
                windows.append(window)
                seen.add(key)
                break
        if len(windows) == target_count:
            return windows

    for segment in ordered:
        duration = _segment_duration(segment)
        text = _clean_text(str(segment.get("text", "")))
        key = (round(float(segment["start"]), 3), round(float(segment["end"]), 3))
        overlaps_existing = any(
            float(window["start"]) < float(segment["end"]) and float(segment["start"]) < float(window["end"])
            for window in windows
        )
        if min_seconds <= duration <= max_seconds and len(text) >= 12 and key not in seen and not overlaps_existing:
            windows.append(_make_window([segment]))
            if len(windows) == target_count:
                return windows

    if len(windows) >= target_count:
        return windows[:target_count]

    for index, segment in enumerate(ordered):
        group = [segment]
        end = float(segment["end"])
        for neighbor in ordered[index + 1 :]:
            if float(neighbor["start"]) - end > 1.0:
                break
            next_end = float(neighbor["end"])
            if next_end - float(group[0]["start"]) > max_seconds:
                break
            group.append(neighbor)
            end = next_end
            duration = end - float(group[0]["start"])
            if duration >= min_seconds:
                window = _make_window(group)
                key = (round(float(window["start"]), 3), round(float(window["end"]), 3))
                if key not in seen:
                    windows.append(window)
                    seen.add(key)
                break
        if len(windows) == target_count:
            return sorted(windows, key=lambda window: float(window["start"]))

    return sorted(windows, key=lambda window: float(window["start"]))[:target_count]


def render_clip(ffmpeg: str, source_video: Path, destination: Path, start: float, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        _clip_filter(),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
    subprocess.run(command, check=True)


def write_training_zip(
    *,
    source_video: Path,
    transcript_json: Path,
    output_dir: Path,
    ffmpeg: str,
    target_count: int = 20,
) -> Path:
    transcript = json.loads(transcript_json.read_text(encoding="utf-8"))
    windows = choose_clip_windows(transcript["segments"], target_count=target_count)
    if len(windows) < target_count:
        raise RuntimeError(f"only found {len(windows)} usable clip windows; expected {target_count}")

    training_dir = output_dir / "training"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    training_dir.mkdir(parents=True)

    manifest = []
    for index, window in enumerate(windows, start=1):
        stem = f"orvo_{index:03d}"
        video_path = training_dir / f"{stem}.mp4"
        render_clip(
            ffmpeg,
            source_video,
            video_path,
            start=max(0.0, float(window["start"]) - 0.15),
            duration=float(window["duration"]) + 0.3,
        )
        caption_path = training_dir / f"{stem}.txt"
        caption_path.write_text(str(window["caption"]) + "\n", encoding="utf-8")
        manifest.append({**window, "stem": stem, "video": str(video_path), "caption_file": str(caption_path)})

    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(output_dir / "training"), "zip", root_dir=training_dir)
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tutorial-style LTX-2.3 character LoRA dataset")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--transcript-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--target-count", type=int, default=20)
    args = parser.parse_args()

    archive = write_training_zip(
        source_video=args.source_video,
        transcript_json=args.transcript_json,
        output_dir=args.output_dir,
        ffmpeg=args.ffmpeg,
        target_count=args.target_count,
    )
    print(json.dumps({"training_zip": str(archive), "target_count": args.target_count}, indent=2))


if __name__ == "__main__":
    main()
