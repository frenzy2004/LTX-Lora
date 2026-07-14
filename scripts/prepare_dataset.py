from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ltx_lora_pilot.dataset import portrait_video_filter, safe_reset_output_directory, select_records


TRIGGER = "chrx9_person"


def normalize(source: Path, destination: Path, start: float, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        portrait_video_filter(),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a neutral video-only fal training archive")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--holdout-count", type=int, default=5)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()

    records = json.loads(args.inventory.read_text(encoding="utf-8"))
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        training, holdout = select_records(
            records,
            selection["training_source_ids"],
            selection["holdout_source_ids"],
            clip_seconds=args.clip_seconds,
        )
        if len(training) != args.train_count or len(holdout) != args.holdout_count:
            raise ValueError(
                f"selection has {len(training)} training and {len(holdout)} holdout sources; "
                f"expected {args.train_count} and {args.holdout_count}"
            )
    else:
        eligible = [
            record
            for record in records
            if record["duration"] >= args.clip_seconds and min(record["width"], record["height"]) >= 480
        ]
        eligible.sort(
            key=lambda record: (-min(record["width"], record["height"]), -record["duration"], record["source_id"])
        )
        needed = args.train_count + args.holdout_count
        if len(eligible) < needed:
            raise RuntimeError(f"need {needed} eligible sources, found {len(eligible)}")
        chosen = eligible[:needed]
        training = chosen[: args.train_count]
        holdout = chosen[args.train_count :]
    train_dir = args.output / "training"
    holdout_dir = args.output / "holdout"
    for directory in (train_dir, holdout_dir):
        safe_reset_output_directory(args.output, directory)

    def render(records_to_render: list[dict], directory: Path, prefix: str) -> list[dict]:
        manifest = []
        for index, record in enumerate(records_to_render, start=1):
            source = Path(record["path"])
            start = max(0.0, (float(record["duration"]) - args.clip_seconds) / 2)
            stem = f"{prefix}_{index:03d}"
            video = directory / f"{stem}.mp4"
            normalize(source, video, start, args.clip_seconds)
            caption = f"{TRIGGER} speaking naturally to the camera, realistic talking-head video"
            (directory / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
            manifest.append({"sample": stem, "source_id": record["source_id"], "duration": args.clip_seconds})
        return manifest

    private_manifest = {
        "training": render(training, train_dir, "train"),
        "holdout": render(holdout, holdout_dir, "holdout"),
    }
    (args.output / "private_manifest.json").write_text(json.dumps(private_manifest, indent=2) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(args.output / "training"), "zip", root_dir=train_dir)
    print(f"created {archive} with {len(training)} training clips and {len(holdout)} held-out clips")


if __name__ == "__main__":
    main()
