from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

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
from run_ltx_a2v_lora import (
    mux_exact_audio,
    upload_files,
    verify_expected_sha256,
)
from run_ltx_a2v_motion import make_balanced_control, valid_frame_count_for_duration


LTX23_PRO_A2V_APPLICATION = "fal-ai/ltx-2.3/audio-to-video"
LTX23_PRO_A2V_QUEUE_URL = "https://queue.fal.run/fal-ai/ltx-2.3/audio-to-video"
PRO_A2V_RATE_PER_SECOND = 0.10


class ProA2VExecutionError(RuntimeError):
    pass


def _secure_url(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{label} must be a secure URL")
    return value


def build_pro_a2v_input(
    *, audio_url: str, image_url: str, prompt: str
) -> dict[str, Any]:
    _secure_url(audio_url, "audio_url")
    _secure_url(image_url, "image_url")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    return {
        "audio_url": audio_url,
        "image_url": image_url,
        "prompt": prompt.strip(),
        "guidance_scale": 9.0,
        "aspect_ratio": "9:16",
    }


def estimate_cost(duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return duration_seconds * PRO_A2V_RATE_PER_SECOND


def submit_inference_once(
    application: str,
    arguments: dict[str, Any],
    key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    if application != LTX23_PRO_A2V_APPLICATION:
        raise ValueError("LTX 2.3 Pro A2V endpoint is fixed")
    if not isinstance(key, str) or not key:
        raise ProA2VExecutionError("Fal key is unavailable")
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
                LTX23_PRO_A2V_QUEUE_URL,
                content=canonical_json_bytes(arguments),
                headers=headers,
            )
    except Exception as exc:
        raise ProA2VExecutionError(
            "provider Pro A2V transport outcome is ambiguous; do not retry"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise ProA2VExecutionError(
            f"provider Pro A2V returned non-success status {response.status_code}"
        )
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise ProA2VExecutionError(
            "provider Pro A2V acknowledgement is malformed; do not retry"
        ) from exc
    request_id = (
        acknowledgement.get("request_id")
        if isinstance(acknowledgement, dict)
        else None
    )
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProA2VExecutionError(
            "provider Pro A2V acknowledgement is malformed; do not retry"
        )
    return {"request_id": request_id}


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
        raise ProA2VExecutionError("audio duration is invalid")
    return duration


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


def make_rgb_reference(video: Path, destination: Path, frames: int) -> None:
    filter_graph = (
        "fps=24,"
        "scale=544:960:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=544:960,"
        "tpad=stop_mode=clone:stop_duration=1,"
        f"trim=end_frame={frames},setpts=N/(24*TB)"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            filter_graph,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            str(destination),
        ],
        check=True,
    )


def start_run(
    *,
    audio: Path,
    expected_audio_sha256: str,
    image: Path,
    prompt: str,
    budget_path: Path,
    key_source: Path,
    state_dir: Path,
    label: str,
    amount: float,
) -> dict[str, Any]:
    for required in (audio, image):
        if not required.is_file():
            raise ProA2VExecutionError(f"missing required input: {required.name}")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "execution.private.json"
    if state_path.exists():
        raise ProA2VExecutionError(
            "execution state already exists; refusing duplicate paid submission"
        )
    verified_audio_sha256 = verify_expected_sha256(audio, expected_audio_sha256)
    duration = _audio_duration(audio)
    if duration < 10.5 or duration > 11.5:
        raise ProA2VExecutionError(
            "audio duration is outside the guarded complete eleven-second target"
        )
    expected_frames = valid_frame_count_for_duration(duration, 24)
    reserve_budget_file(budget_path, label, amount)
    state: dict[str, Any] = {
        "application": LTX23_PRO_A2V_APPLICATION,
        "budget_label": label,
        "reserved_amount_usd": amount,
        "phase": "reserved",
        "created_at_utc": utc_now(),
        "prompt": prompt,
        "audio_duration_seconds": duration,
        "expected_normalized_frames": expected_frames,
        "estimated_provider_cost_usd": round(estimate_cost(duration), 6),
        "audio_sha256": verified_audio_sha256,
        "image_sha256": sha256_file(image).upper(),
        "audio_path": str(audio.resolve()),
    }
    atomic_write_json(state_path, state)
    key = extract_unique_fal_key(key_source)
    try:
        urls = upload_files({"audio": audio, "image": image}, key)
    except Exception:
        state["phase"] = "upload_failed_unsubmitted"
        state["failed_at_utc"] = utc_now()
        atomic_write_json(state_path, state)
        raise
    arguments = build_pro_a2v_input(
        audio_url=urls["audio"], image_url=urls["image"], prompt=prompt
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
        LTX23_PRO_A2V_APPLICATION, arguments, key
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
        "audio_duration_seconds": duration,
        "expected_normalized_frames": expected_frames,
    }


def _record_failed_result(
    state: dict[str, Any], state_path: Path, budget_path: Path, outcome: str
) -> dict[str, Any]:
    state["phase"] = "failed_pending_billing_verification"
    state["completed_at_utc"] = utc_now()
    atomic_write_json(state_path, state)
    update_budget_entry(
        budget_path,
        state["budget_label"],
        "failed_pending_billing_verification",
        completed_at_utc=state["completed_at_utc"],
        provider_outcome=outcome,
    )
    return {"phase": state["phase"], "result_available": False}


def monitor_run(
    *, state_dir: Path, budget_path: Path, key_source: Path
) -> dict[str, Any]:
    state_path = state_dir / "execution.private.json"
    if not state_path.is_file():
        raise ProA2VExecutionError("execution state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request_id = state.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProA2VExecutionError("execution has no submitted provider request")
    key = extract_unique_fal_key(key_source)
    import fal_client

    client = fal_client.SyncClient(key=key, default_timeout=180.0)
    status = client.status(LTX23_PRO_A2V_APPLICATION, request_id, with_logs=False)
    status_name = type(status).__name__.lower()
    state["last_provider_status"] = status_name
    state["last_checked_at_utc"] = utc_now()
    atomic_write_json(state_path, state)
    if status_name != "completed":
        return {"phase": state["phase"], "provider_status": status_name}
    if getattr(status, "error", None):
        output = _record_failed_result(
            state, state_path, budget_path, "provider_reported_error"
        )
        output["provider_status"] = status_name
        return output
    try:
        result = client.result(LTX23_PRO_A2V_APPLICATION, request_id)
    except Exception as exc:
        state["result_fetch_error_type"] = type(exc).__name__
        output = _record_failed_result(
            state,
            state_path,
            budget_path,
            "completed_without_retrievable_result",
        )
        output["provider_status"] = status_name
        return output
    if not isinstance(result, dict):
        raise ProA2VExecutionError("provider Pro A2V result has an unexpected shape")
    video_value = result.get("video")
    video_url = video_value.get("url") if isinstance(video_value, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith("https://"):
        raise ProA2VExecutionError("completed Pro A2V result has no downloadable video")
    atomic_write_json(state_dir / "result.private.json", result)
    provider_video = state_dir / "pro_a2v_provider_audio.mp4"
    exact_audio = state_dir / "pro_a2v_exact_audio.mp4"
    rgb_reference = state_dir / "pro_a2v_rgb_reference_544x960_24fps.mp4"
    control = state_dir / "pro_a2v_balanced_control_544x960_24fps.mp4"
    _download_video(video_url, provider_video)
    mux_exact_audio(provider_video, Path(state["audio_path"]), exact_audio)
    make_rgb_reference(
        provider_video, rgb_reference, int(state["expected_normalized_frames"])
    )
    make_balanced_control(
        rgb_reference, control, int(state["expected_normalized_frames"])
    )
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
            "rgb_reference": {
                "size_bytes": rgb_reference.stat().st_size,
                "sha256": sha256_file(rgb_reference).upper(),
            },
            "balanced_control": {
                "size_bytes": control.stat().st_size,
                "sha256": sha256_file(control).upper(),
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
        "outputs": [
            provider_video.name,
            exact_audio.name,
            rgb_reference.name,
            control.name,
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded one-shot latest LTX 2.3 Pro audio-to-video"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    start = subs.add_parser("start")
    start.add_argument("--audio", type=Path, required=True)
    start.add_argument("--expected-audio-sha256", required=True)
    start.add_argument("--image", type=Path, required=True)
    start.add_argument("--prompt", required=True)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--key-source", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--amount", type=float, default=1.10)
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
            prompt=args.prompt,
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
