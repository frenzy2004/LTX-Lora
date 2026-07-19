from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


class DatasetError(RuntimeError):
    """Raised when dataset evidence violates an immutable run constraint."""


class BudgetExceeded(RuntimeError):
    """Raised before a provider request can exceed the authorised cap."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {path}")
    return value


def load_authoritative_groups(
    dataset_record_path: Path, provenance_path: Path
) -> dict[str, list[dict[str, Any]]]:
    """Bind approved group IDs to their private provenance without widening scope."""

    record = _load_json(dataset_record_path)
    provenance = _load_json(provenance_path)
    validation = record.get("quality_validation", {})
    if validation.get("status") != "valid":
        raise DatasetError("The reviewed dataset record is not marked valid")

    accepted = {
        "train": validation.get("accepted_train_group_ids", []),
        "holdout": validation.get("accepted_holdout_group_ids", []),
    }
    if len(accepted["train"]) != record.get("counts", {}).get("train_groups"):
        raise DatasetError("Approved training count does not match the dataset record")
    if len(accepted["holdout"]) != record.get("counts", {}).get("holdout_groups"):
        raise DatasetError("Approved holdout count does not match the dataset record")

    index: dict[str, dict[str, Any]] = {}
    for group in provenance.get("groups", []):
        group_id = group.get("opaque_group_id")
        if not group_id or group_id in index:
            raise DatasetError("Provenance contains a missing or duplicate group ID")
        index[group_id] = group

    result: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    for split, ids in accepted.items():
        if len(ids) != len(set(ids)):
            raise DatasetError(f"Approved {split} IDs contain duplicates")
        for group_id in ids:
            if group_id not in index:
                raise DatasetError(f"Approved group is missing from provenance: {group_id}")
            group = copy.deepcopy(index[group_id])
            if group.get("split") != split:
                raise DatasetError(
                    f"Approved {split} group has conflicting provenance split: {group_id}"
                )
            result[split].append(group)

    assert_split_isolation(result["train"], result["holdout"])
    return result


def assert_split_isolation(
    train: list[dict[str, Any]], holdout: list[dict[str, Any]]
) -> None:
    checks = (
        ("opaque_source_asset_id", "source asset"),
        ("opaque_source_session_id", "source session"),
        ("opaque_location_id", "location"),
    )
    for field, label in checks:
        train_values = {item.get(field) for item in train if item.get(field)}
        holdout_values = {item.get(field) for item in holdout if item.get(field)}
        collisions = train_values & holdout_values
        if collisions:
            raise DatasetError(
                f"Train/holdout {label} collision: {', '.join(sorted(collisions))}"
            )


def build_pair_paths(root: Path, sample_id: str) -> dict[str, Path]:
    if not sample_id or any(char in sample_id for char in "\\/:"):
        raise DatasetError(f"Unsafe sample ID: {sample_id!r}")
    return {
        "control": root / f"{sample_id}_start.mp4",
        "target": root / f"{sample_id}_end.mp4",
        "caption": root / f"{sample_id}.txt",
    }


_CANNY_PROFILES: dict[str, tuple[str, str, str | None]] = {
    # FFmpeg defaults: useful as the dense end of the visual audit.
    "dense": ("0.078431", "0.196078", "0.60"),
    # Midpoint between FFmpeg defaults and the official OpenCV thresholds.
    "balanced": ("0.196078", "0.392157", "0.35"),
    # Official LTX compute_reference.py uses OpenCV Canny(100, 200).
    "official_sparse": ("0.392157", "0.784314", None),
}


def build_canny_filter(profile: str) -> str:
    try:
        low, high, blur = _CANNY_PROFILES[profile]
    except KeyError as exc:
        raise DatasetError(f"Unknown Canny profile: {profile}") from exc

    parts = ["format=gray"]
    if blur is not None:
        parts.append(f"gblur=sigma={blur}")
    parts.append(
        f"edgedetect=low={low}:high={high}:mode=wires:planes=y"
    )
    parts.append("format=yuv420p")
    return ",".join(parts)


def build_crop_filter(crop: dict[str, Any]) -> str:
    coordinate_space = crop.get("coordinate_space")
    if coordinate_space not in {"display_pixels", "display_pixels_after_rotation"}:
        raise DatasetError(f"Unsupported crop coordinate space: {coordinate_space!r}")
    required = ("x", "y", "width", "height", "output_width", "output_height")
    try:
        x, y, width, height, output_width, output_height = (
            int(crop[field]) for field in required
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError("Crop metadata is missing an integer geometry field") from exc
    if min(x, y) < 0 or min(width, height, output_width, output_height) <= 0:
        raise DatasetError("Crop geometry must be positive and within display space")
    source_width = int(crop.get("source_display_width", 0))
    source_height = int(crop.get("source_display_height", 0))
    if source_width and x + width > source_width:
        raise DatasetError("Crop extends beyond display width")
    if source_height and y + height > source_height:
        raise DatasetError("Crop extends beyond display height")
    if (output_width, output_height) != (544, 960):
        raise DatasetError("This run requires a 544x960 output crop")
    return (
        f"crop={width}:{height}:{x}:{y},"
        f"scale={output_width}:{output_height}:flags=lanczos,"
        "fps=24,format=yuv420p"
    )


def discover_speech_windows(
    words: list[dict[str, Any]],
    *,
    source_duration: float,
    window_seconds: float = 89 / 24,
    minimum_words: int = 4,
    minimum_speech_coverage_ratio: float = 0.30,
) -> list[dict[str, Any]]:
    """Rank fixed-length windows from Whisper word timestamps."""

    clean_words: list[dict[str, Any]] = []
    for word in words:
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or start < 0 or start >= source_duration:
            continue
        clean_words.append(
            {
                "start": start,
                "end": min(end, source_duration),
                "word": str(word.get("word", "")).strip(),
            }
        )
    clean_words.sort(key=lambda item: (item["start"], item["end"]))
    if not clean_words or source_duration < window_seconds:
        return []

    maximum_start = source_duration - window_seconds
    candidate_starts = {
        round(max(0.0, min(word["start"] - 0.15, maximum_start)), 6)
        for word in clean_words
    }
    ranked: list[dict[str, Any]] = []
    for start in sorted(candidate_starts):
        end = start + window_seconds
        included = [
            word
            for word in clean_words
            if word["start"] < end and word["end"] > start
        ]
        if len(included) < minimum_words:
            continue
        speech_seconds = sum(
            max(0.0, min(word["end"], end) - max(word["start"], start))
            for word in included
        )
        coverage = min(1.0, speech_seconds / window_seconds)
        if coverage < minimum_speech_coverage_ratio:
            continue
        gaps = []
        cursor = start
        for word in included:
            clipped_start = max(start, word["start"])
            clipped_end = min(end, word["end"])
            gaps.append(max(0.0, clipped_start - cursor))
            cursor = max(cursor, clipped_end)
        gaps.append(max(0.0, end - cursor))
        longest_gap = max(gaps)
        score = (
            len(included) * 2.0
            + coverage * 8.0
            - longest_gap * 1.5
            - abs(included[0]["start"] - start) * 0.15
        )
        ranked.append(
            {
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "word_count": len(included),
                "speech_seconds": round(speech_seconds, 6),
                "speech_coverage_ratio": round(coverage, 6),
                "longest_gap_seconds": round(longest_gap, 6),
                "score": round(score, 6),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (-item["score"], item["start_seconds"]),
    )


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def select_nonoverlapping_windows(
    candidates: list[dict[str, Any]],
    *,
    blocked_intervals: list[tuple[float, float]] | None = None,
    maximum: int,
    minimum_gap_seconds: float = 0.20,
) -> list[dict[str, Any]]:
    blocked_intervals = blocked_intervals or []
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-float(item["score"]), float(item["start_seconds"])),
    ):
        start = float(candidate["start_seconds"])
        end = float(candidate["end_seconds"])
        if any(_overlaps(start, end, b_start, b_end) for b_start, b_end in blocked_intervals):
            continue
        if any(
            _overlaps(
                start - minimum_gap_seconds,
                end + minimum_gap_seconds,
                float(item["start_seconds"]),
                float(item["end_seconds"]),
            )
            for item in selected
        ):
            continue
        selected.append(copy.deepcopy(candidate))
        if len(selected) >= maximum:
            break
    return sorted(selected, key=lambda item: float(item["start_seconds"]))


def validate_video_probe(
    probe: dict[str, Any], *, require_silent: bool = True
) -> None:
    streams = probe.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise DatasetError(f"Expected exactly one video stream, found {len(videos)}")
    if require_silent and audios:
        raise DatasetError("Control video must not contain an audio stream")

    video = videos[0]
    if (video.get("width"), video.get("height")) != (544, 960):
        raise DatasetError(
            f"Expected 544x960, found {video.get('width')}x{video.get('height')}"
        )
    if video.get("pix_fmt") != "yuv420p":
        raise DatasetError(f"Expected yuv420p, found {video.get('pix_fmt')}")

    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate")
    try:
        rate = Fraction(rate_text)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise DatasetError(f"Invalid frame rate: {rate_text!r}") from exc
    if rate != Fraction(24, 1):
        raise DatasetError(f"Expected 24 fps, found {rate}")

    try:
        frames = int(video.get("nb_frames"))
    except (TypeError, ValueError) as exc:
        raise DatasetError("Video probe does not report an exact frame count") from exc
    if frames != 89:
        raise DatasetError(f"Expected 89 frames, found {frames}")

    duration_text = video.get("duration") or probe.get("format", {}).get("duration")
    try:
        duration = float(duration_text)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Invalid duration: {duration_text!r}") from exc
    expected_duration = 89 / 24
    if abs(duration - expected_duration) > 1 / 24:
        raise DatasetError(
            f"Expected approximately {expected_duration:.6f}s, found {duration:.6f}s"
        )


def reserve_budget(
    budget: dict[str, Any], label: str, amount: float
) -> dict[str, Any]:
    """Return a new ledger with a conservative pre-submission reservation."""

    updated = copy.deepcopy(budget)
    amount_decimal = Decimal(str(amount))
    current = Decimal(str(updated.get("incremental_accounted_or_reserved", 0)))
    absolute_stop = Decimal(str(updated["incremental_absolute_stop"]))
    if amount_decimal <= 0:
        raise ValueError("Reservation amount must be positive")
    if current + amount_decimal > absolute_stop:
        raise BudgetExceeded(
            f"Reservation would exceed ${absolute_stop:.2f} incremental cap"
        )

    new_total = current + amount_decimal
    updated["incremental_accounted_or_reserved"] = float(new_total)
    updated["incremental_remaining_absolute"] = float(absolute_stop - new_total)
    if "incremental_normal_cap" in updated:
        normal_cap = Decimal(str(updated["incremental_normal_cap"]))
        updated["incremental_remaining_normal_cap"] = float(normal_cap - new_total)
    updated.setdefault("entries", []).append(
        {
            "label": label,
            "amount_usd": float(amount_decimal),
            "status": "reserved",
            "reserved_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )
    return updated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hash_manifest(
    items: Iterable[tuple[str, str, Path]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for split, sample_id, path in items:
        if split not in {"train", "holdout"}:
            raise DatasetError(f"Unknown split: {split}")
        if not path.is_file():
            raise DatasetError(f"Missing file: {path}")
        manifest.append(
            {
                "split": split,
                "sample_id": sample_id,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return manifest


def assert_no_hash_collisions(manifest: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        by_hash.setdefault(item["sha256"], []).append(item)
    for digest, items in by_hash.items():
        splits = {item["split"] for item in items}
        if len(splits) > 1:
            samples = ", ".join(item["sample_id"] for item in items)
            raise DatasetError(
                f"Cross-split hash collision {digest[:12]} among: {samples}"
            )
