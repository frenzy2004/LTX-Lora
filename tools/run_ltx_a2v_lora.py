from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import httpx

from run_ic_lora_provider import (
    atomic_write_json,
    canonical_json_bytes,
    extract_unique_fal_key,
    reserve_budget_file,
    sha256_file,
    update_budget_entry,
    utc_now,
)


LTX_A2V_LORA_APPLICATION = "fal-ai/ltx-2.3-quality/audio-to-video/lora"
LTX_A2V_LORA_QUEUE_URL = (
    "https://queue.fal.run/fal-ai/ltx-2.3-quality/audio-to-video/lora"
)
QUALITY_LORA_RATE_PER_MEGAPIXEL_FRAME = 0.0027075
NEGATIVE_PROMPT = (
    "AI-generated, obvious AI video, CGI, 3D render, cartoon, illustration, "
    "waxy skin, plastic skin, over-smoothed face, identity drift, changing face, "
    "asymmetrical eyes, deformed mouth, mismatched lip motion, extra teeth, "
    "warped beard, flicker, temporal jitter, motion smear, ghosting, subtitles, "
    "captions, logos, watermarks"
)


class A2VLoRAExecutionError(RuntimeError):
    pass


def valid_frame_count_for_duration(duration_seconds: float, fps: int = 24) -> int:
    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration_seconds and fps must be positive")
    raw_frames = math.ceil(duration_seconds * fps)
    return max(1, 8 * math.ceil((raw_frames - 1) / 8) + 1)


def estimate_cost(width: int, height: int, frames: int) -> float:
    if min(width, height, frames) <= 0:
        raise ValueError("width, height, and frames must be positive")
    return (
        width
        * height
        * frames
        / 1_000_000
        * QUALITY_LORA_RATE_PER_MEGAPIXEL_FRAME
    )


def verify_expected_sha256(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise A2VLoRAExecutionError("required audio file is unavailable")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("expected SHA-256 must be hexadecimal") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != expected_sha256.upper():
        raise A2VLoRAExecutionError("audio hash mismatch; refusing paid submission")
    return digest


def _secure_url(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{label} must be a secure URL")
    return value


def build_a2v_lora_input(
    *,
    audio_url: str,
    image_url: str,
    lora_url: str,
    prompt: str,
    seed: int,
    lora_scale: float,
) -> dict[str, Any]:
    _secure_url(audio_url, "audio_url")
    _secure_url(image_url, "image_url")
    _secure_url(lora_url, "lora_url")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    return {
        "prompt": prompt.strip(),
        "audio_url": audio_url,
        "image_url": image_url,
        "match_audio_length": True,
        "resolution": "auto",
        "frames_per_second": 24,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "generate_audio": True,
        "image_strength": 0.90,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": int(seed),
        "enable_prompt_expansion": False,
        "enable_safety_checker": True,
        "video_quality": "maximum",
        "video_write_mode": "balanced",
        "sync_mode": False,
        "loras": [
            {
                "path": lora_url,
                "scale": float(lora_scale),
                "transformer": "both",
            }
        ],
    }


def submit_inference_once(
    application: str,
    arguments: dict[str, Any],
    key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    if application != LTX_A2V_LORA_APPLICATION:
        raise ValueError("LTX quality A2V-LoRA endpoint is fixed")
    if not isinstance(key, str) or not key:
        raise A2VLoRAExecutionError("Fal key is unavailable")
    selected_transport: httpx.BaseTransport = transport or httpx.HTTPTransport(
        retries=0, verify=True, trust_env=False, http1=True, http2=False
    )
    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Fal-No-Retry": "1",
        "x-app-fal-disable-fallback": "true",
        "X-Fal-Store-IO": "0",
    }
    try:
        with httpx.Client(
            transport=selected_transport,
            follow_redirects=False,
            trust_env=False,
            http1=True,
            http2=False,
            timeout=120.0,
        ) as client:
            response = client.post(
                LTX_A2V_LORA_QUEUE_URL,
                content=canonical_json_bytes(arguments),
                headers=headers,
            )
    except Exception as exc:
        raise A2VLoRAExecutionError(
            "provider A2V-LoRA transport outcome is ambiguous; do not retry"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise A2VLoRAExecutionError(
            f"provider A2V-LoRA returned non-success status {response.status_code}"
        )
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise A2VLoRAExecutionError(
            "provider A2V-LoRA acknowledgement is malformed; do not retry"
        ) from exc
    request_id = (
        acknowledgement.get("request_id")
        if isinstance(acknowledgement, dict)
        else None
    )
    if not isinstance(request_id, str) or not request_id.strip():
        raise A2VLoRAExecutionError(
            "provider A2V-LoRA acknowledgement is malformed; do not retry"
        )
    return {"request_id": request_id}


def upload_files(
    paths: dict[str, Path],
    key: str,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    for path in paths.values():
        if not path.is_file():
            raise A2VLoRAExecutionError("required local A2V-LoRA input is unavailable")
    if client_factory is None:
        import fal_client

        client_factory = lambda selected_key: fal_client.SyncClient(
            key=selected_key, default_timeout=900.0
        )
    client = client_factory(key)
    urls: dict[str, str] = {}
    for label, path in paths.items():
        url = client.upload_file(path)
        if not isinstance(url, str) or not url.startswith("https://"):
            raise A2VLoRAExecutionError(
                f"provider upload for {label} did not return a secure URL"
            )
        urls[label] = url
    return urls


def _audio_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(completed.stdout.strip())
    if duration <= 0:
        raise A2VLoRAExecutionError("audio duration is invalid")
    return duration


def mux_exact_audio(video: Path, audio: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def _download_video(url: str, destination: Path) -> None:
    _secure_url(url, "video_url")
    temporary = destination.with_suffix(destination.suffix + ".part")
    transport = httpx.HTTPTransport(retries=0, verify=True, trust_env=False)
    with httpx.Client(
        transport=transport,
        follow_redirects=True,
        trust_env=False,
        timeout=900.0,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
    os.replace(temporary, destination)


def start_run(
    *,
    audio: Path,
    expected_audio_sha256: str,
    image: Path,
    lora: Path,
    prompt: str,
    seed: int,
    lora_scale: float,
    budget_path: Path,
    key_source: Path,
    state_dir: Path,
    label: str,
    amount: float,
) -> dict[str, Any]:
    for required in (audio, image, lora):
        if not required.is_file():
            raise A2VLoRAExecutionError(f"missing required input: {required.name}")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "execution.private.json"
    if state_path.exists():
        raise A2VLoRAExecutionError(
            "execution state already exists; refusing duplicate paid submission"
        )
    verified_audio_sha256 = verify_expected_sha256(audio, expected_audio_sha256)
    duration = _audio_duration(audio)
    if duration < 10.5 or duration > 11.5:
        raise A2VLoRAExecutionError(
            "audio duration is outside the guarded complete eleven-second target"
        )
    expected_frames = valid_frame_count_for_duration(duration, 24)
    reserve_budget_file(budget_path, label, amount)
    state: dict[str, Any] = {
        "application": LTX_A2V_LORA_APPLICATION,
        "budget_label": label,
        "reserved_amount_usd": amount,
        "phase": "reserved",
        "created_at_utc": utc_now(),
        "prompt": prompt,
        "seed": seed,
        "lora_scale": lora_scale,
        "audio_duration_seconds": duration,
        "expected_normalized_frames": expected_frames,
        "audio_sha256": verified_audio_sha256,
        "image_sha256": sha256_file(image).upper(),
        "lora_sha256": sha256_file(lora).upper(),
        "audio_path": str(audio.resolve()),
    }
    atomic_write_json(state_path, state)

    key = extract_unique_fal_key(key_source)
    try:
        urls = upload_files(
            {"audio": audio, "image": image, "lora": lora}, key
        )
    except Exception:
        state["phase"] = "upload_failed_unsubmitted"
        state["failed_at_utc"] = utc_now()
        atomic_write_json(state_path, state)
        raise
    arguments = build_a2v_lora_input(
        audio_url=urls["audio"],
        image_url=urls["image"],
        lora_url=urls["lora"],
        prompt=prompt,
        seed=seed,
        lora_scale=lora_scale,
    )
    state.update(
        {
            "phase": "uploaded",
            "request_body_sha256": hashlib.sha256(
                canonical_json_bytes(arguments)
            ).hexdigest().upper(),
            "uploaded_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    acknowledgement = submit_inference_once(
        LTX_A2V_LORA_APPLICATION, arguments, key
    )
    state.update(
        {
            "phase": "submitted",
            "request_id": acknowledgement["request_id"],
            "submitted_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    update_budget_entry(
        budget_path, label, "submitted", submitted_at_utc=state["submitted_at_utc"]
    )
    return {
        "phase": "submitted",
        "reserved_amount_usd": amount,
        "seed": seed,
        "audio_duration_seconds": duration,
        "expected_normalized_frames": expected_frames,
    }


def monitor_run(
    *, state_dir: Path, budget_path: Path, key_source: Path
) -> dict[str, Any]:
    state_path = state_dir / "execution.private.json"
    if not state_path.is_file():
        raise A2VLoRAExecutionError("execution state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request_id = state.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise A2VLoRAExecutionError("execution has no submitted provider request")
    key = extract_unique_fal_key(key_source)
    import fal_client

    client = fal_client.SyncClient(key=key, default_timeout=180.0)
    status = client.status(LTX_A2V_LORA_APPLICATION, request_id, with_logs=False)
    status_name = type(status).__name__.lower()
    state["last_provider_status"] = status_name
    state["last_checked_at_utc"] = utc_now()
    atomic_write_json(state_path, state)
    if status_name != "completed":
        return {"phase": state["phase"], "provider_status": status_name}
    if getattr(status, "error", None):
        state["phase"] = "failed_pending_billing_verification"
        state["completed_at_utc"] = utc_now()
        atomic_write_json(state_path, state)
        update_budget_entry(
            budget_path,
            state["budget_label"],
            "failed_pending_billing_verification",
            completed_at_utc=state["completed_at_utc"],
        )
        return {"phase": state["phase"], "provider_status": status_name}
    try:
        result = client.result(LTX_A2V_LORA_APPLICATION, request_id)
    except Exception as exc:
        state["phase"] = "failed_pending_billing_verification"
        state["completed_at_utc"] = utc_now()
        state["result_fetch_error_type"] = type(exc).__name__
        atomic_write_json(state_path, state)
        update_budget_entry(
            budget_path,
            state["budget_label"],
            "failed_pending_billing_verification",
            completed_at_utc=state["completed_at_utc"],
            provider_outcome="completed_without_retrievable_result",
        )
        return {
            "phase": state["phase"],
            "provider_status": status_name,
            "result_available": False,
        }
    if not isinstance(result, dict):
        raise A2VLoRAExecutionError("provider A2V-LoRA result has an unexpected shape")
    video_value = result.get("video")
    video_url = video_value.get("url") if isinstance(video_value, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith("https://"):
        raise A2VLoRAExecutionError(
            "completed A2V-LoRA result has no downloadable video"
        )
    atomic_write_json(state_dir / "result.private.json", result)
    provider_video = state_dir / "a2v_lora_provider_audio.mp4"
    exact_audio = state_dir / "a2v_lora_exact_audio.mp4"
    _download_video(video_url, provider_video)
    mux_exact_audio(provider_video, Path(state["audio_path"]), exact_audio)
    state.update(
        {
            "phase": "completed",
            "completed_at_utc": utc_now(),
            "provider_video": {
                "size_bytes": provider_video.stat().st_size,
                "sha256": sha256_file(provider_video).upper(),
            },
            "exact_audio_video": {
                "size_bytes": exact_audio.stat().st_size,
                "sha256": sha256_file(exact_audio).upper(),
            },
        }
    )
    atomic_write_json(state_path, state)
    update_budget_entry(
        budget_path,
        state["budget_label"],
        "charged_expected",
        completed_at_utc=state["completed_at_utc"],
    )
    return {
        "phase": "completed",
        "provider_status": status_name,
        "outputs": [provider_video.name, exact_audio.name],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded one-shot latest LTX Quality A2V-LoRA inference"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    start = subs.add_parser("start")
    start.add_argument("--audio", type=Path, required=True)
    start.add_argument("--expected-audio-sha256", required=True)
    start.add_argument("--image", type=Path, required=True)
    start.add_argument("--lora", type=Path, required=True)
    start.add_argument("--prompt", required=True)
    start.add_argument("--seed", type=int, required=True)
    start.add_argument("--lora-scale", type=float, default=1.0)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--key-source", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--amount", type=float, default=0.40)
    monitor = subs.add_parser("monitor")
    monitor.add_argument("--budget", type=Path, required=True)
    monitor.add_argument("--key-source", type=Path, required=True)
    monitor.add_argument("--state-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "start":
        output = start_run(
            audio=args.audio,
            expected_audio_sha256=args.expected_audio_sha256,
            image=args.image,
            lora=args.lora,
            prompt=args.prompt,
            seed=args.seed,
            lora_scale=args.lora_scale,
            budget_path=args.budget,
            key_source=args.key_source,
            state_dir=args.state_dir,
            label=args.label,
            amount=args.amount,
        )
    else:
        output = monitor_run(
            state_dir=args.state_dir,
            budget_path=args.budget,
            key_source=args.key_source,
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
