from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from run_ic_lora_provider import (
    ProviderExecutionError,
    atomic_write_json,
    canonical_json_bytes,
    extract_unique_fal_key,
    release_unsubmitted_budget,
    reserve_budget_file,
    sha256_file,
    update_budget_entry,
    utc_now,
)
from run_ltx_a2v_lora import verify_expected_sha256


LTX_V2V_APPLICATION = "fal-ai/ltx-2.3-quality/reference-video-to-video/lora"
LTX_V2V_QUEUE_URL = (
    "https://queue.fal.run/fal-ai/ltx-2.3-quality/reference-video-to-video/lora"
)
QUALITY_RATE_PER_MEGAPIXEL_FRAME = 0.0027075
NEGATIVE_PROMPT = (
    "AI-generated, obvious AI video, CGI, 3D render, cartoon, illustration, "
    "waxy skin, plastic skin, over-smoothed face, identity drift, changing face, "
    "asymmetrical eyes, deformed mouth, mismatched lip motion, extra teeth, "
    "warped beard, flicker, temporal jitter, frame interpolation artifacts, "
    "motion smear, ghosting, duplicate limbs, malformed hands, incorrect fingers, "
    "subtitles, captions, logos, watermarks, overexposure, underexposure, "
    "oversaturation, cinematic color grading, shallow fake depth of field"
)


class InferenceExecutionError(RuntimeError):
    pass


def estimate_cost(width: int, height: int, frames: int) -> float:
    if min(width, height, frames) <= 0:
        raise ValueError("width, height, and frames must be positive")
    return width * height * frames / 1_000_000 * QUALITY_RATE_PER_MEGAPIXEL_FRAME


def _secure_url(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{label} must be a secure URL")
    return value


def build_inference_input(
    *,
    video_url: str,
    control_video_url: str,
    lora_url: str,
    prompt: str,
    seed: int,
    lora_scale: float,
    image_url: str | None = None,
    num_frames: int = 89,
    preserve_original_video: bool = False,
    strength: float = 1.0,
    video_strength: float = 1.0,
) -> dict[str, Any]:
    _secure_url(video_url, "video_url")
    _secure_url(control_video_url, "control_video_url")
    _secure_url(lora_url, "lora_url")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    if not 0.0 < lora_scale <= 2.0:
        raise ValueError("lora_scale must be within (0, 2]")
    if num_frames <= 0 or num_frames % 8 != 1:
        raise ValueError("num_frames must satisfy the LTX 8n+1 constraint")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")
    if not 0.0 <= video_strength <= 1.0:
        raise ValueError("video_strength must be within [0, 1]")
    body: dict[str, Any] = {
        "prompt": prompt.strip(),
        "video_url": video_url,
        "control_video_url": control_video_url,
        "skip_control_preprocess": True,
        "preserve_original_video": bool(preserve_original_video),
        "video_strength": float(video_strength),
        "strength": float(strength),
        "num_frames": int(num_frames),
        "resolution": "auto",
        "frames_per_second": 24,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "generate_audio": False,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": int(seed),
        "enable_prompt_expansion": False,
        "enable_safety_checker": True,
        "video_quality": "maximum",
        "video_write_mode": "balanced",
        "sync_mode": False,
        "loras": [
            {"path": lora_url, "scale": float(lora_scale), "transformer": "both"}
        ],
    }
    if image_url is not None:
        body["image_url"] = _secure_url(image_url, "image_url")
    return body


def submit_inference_once(
    application: str,
    arguments: dict[str, Any],
    key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    if application != LTX_V2V_APPLICATION:
        raise ValueError("LTX quality V2V LoRA endpoint is fixed")
    if not isinstance(key, str) or not key:
        raise InferenceExecutionError("Fal key is unavailable")
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
                LTX_V2V_QUEUE_URL,
                content=canonical_json_bytes(arguments),
                headers=headers,
            )
    except Exception as exc:
        raise InferenceExecutionError(
            "provider inference transport outcome is ambiguous; do not retry"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise InferenceExecutionError(
            f"provider inference returned non-success status {response.status_code}"
        )
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise InferenceExecutionError(
            "provider inference acknowledgement is malformed; do not retry"
        ) from exc
    request_id = acknowledgement.get("request_id") if isinstance(acknowledgement, dict) else None
    if not isinstance(request_id, str) or not request_id.strip():
        raise InferenceExecutionError(
            "provider inference acknowledgement is malformed; do not retry"
        )
    return {"request_id": request_id}


def upload_files(
    paths: dict[str, Path],
    key: str,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    if not paths:
        raise InferenceExecutionError("no provider upload inputs were supplied")
    for path in paths.values():
        if not path.is_file():
            raise InferenceExecutionError("required local inference input is unavailable")
    if client_factory is None:
        import fal_client

        client_factory = lambda value: fal_client.SyncClient(
            key=value, default_timeout=600.0
        )
    client = client_factory(key)
    uploaded: dict[str, str] = {}
    for label, path in paths.items():
        url = client.upload_file(path)
        if not isinstance(url, str) or not url.startswith("https://"):
            raise InferenceExecutionError(
                "provider upload did not return a secure URL"
            )
        uploaded[label] = url
    return uploaded


def upload_file(path: Path, key: str) -> str:
    return upload_files({"file": path}, key)["file"]


def start_run(
    *,
    video: Path,
    control: Path,
    lora: Path,
    audio: Path,
    expected_audio_sha256: str,
    anchor: Path | None,
    prompt: str,
    seed: int,
    lora_scale: float,
    num_frames: int,
    preserve_original_video: bool,
    strength: float,
    video_strength: float,
    budget_path: Path,
    key_source: Path,
    state_dir: Path,
    label: str,
    amount: float,
) -> dict[str, Any]:
    for required in (video, control, lora, audio):
        if not required.is_file():
            raise InferenceExecutionError(f"missing required input: {required.name}")
    if anchor is not None and not anchor.is_file():
        raise InferenceExecutionError("optional anchor input is unavailable")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "execution.private.json"
    if state_path.exists():
        raise InferenceExecutionError("execution state already exists; refusing duplicate submit")
    verified_audio_sha256 = verify_expected_sha256(
        audio, expected_audio_sha256
    )
    reserve_budget_file(budget_path, label, amount)
    state: dict[str, Any] = {
        "application": LTX_V2V_APPLICATION,
        "budget_label": label,
        "reserved_amount_usd": amount,
        "phase": "reserved",
        "created_at_utc": utc_now(),
        "prompt": prompt,
        "seed": seed,
        "lora_scale": lora_scale,
        "num_frames": num_frames,
        "preserve_original_video": preserve_original_video,
        "strength": strength,
        "video_strength": video_strength,
        "video_sha256": sha256_file(video),
        "control_sha256": sha256_file(control),
        "lora_sha256": sha256_file(lora),
        "audio_sha256": verified_audio_sha256,
        "audio_path": str(audio.resolve()),
    }
    if anchor is not None:
        state["anchor_sha256"] = sha256_file(anchor)
    atomic_write_json(state_path, state)

    key = extract_unique_fal_key(key_source)
    upload_inputs = {"video": video, "control": control, "lora": lora}
    if anchor is not None:
        upload_inputs["anchor"] = anchor
    try:
        uploaded = upload_files(upload_inputs, key)
    except Exception as exc:
        state["phase"] = "upload_failed_unsubmitted"
        state["failed_at_utc"] = utc_now()
        state["upload_error_type"] = type(exc).__name__
        atomic_write_json(state_path, state)
        release_unsubmitted_budget(
            budget_path,
            label,
            "provider input upload failed before request acknowledgement; "
            "execution state contains no request_id or submitted_at_utc",
        )
        raise InferenceExecutionError(
            "provider input upload failed before submission; reservation released"
        ) from exc
    control_url = uploaded["control"]
    lora_url = uploaded["lora"]
    anchor_url = uploaded.get("anchor")
    arguments = build_inference_input(
        video_url=uploaded["video"],
        control_video_url=control_url,
        lora_url=lora_url,
        prompt=prompt,
        seed=seed,
        lora_scale=lora_scale,
        image_url=anchor_url,
        num_frames=num_frames,
        preserve_original_video=preserve_original_video,
        strength=strength,
        video_strength=video_strength,
    )
    state.update(
        {
            "phase": "uploaded",
            "request_body_sha256": hashlib.sha256(
                canonical_json_bytes(arguments)
            ).hexdigest(),
            "uploaded_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    acknowledgement = submit_inference_once(LTX_V2V_APPLICATION, arguments, key)
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
        "lora_scale": lora_scale,
    }


def _download_video(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    transport = httpx.HTTPTransport(retries=0, verify=True, trust_env=False)
    with httpx.Client(
        transport=transport,
        follow_redirects=True,
        trust_env=False,
        timeout=600.0,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
    os.replace(temporary, destination)


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


def monitor_run(
    *, state_dir: Path, budget_path: Path, key_source: Path
) -> dict[str, Any]:
    state_path = state_dir / "execution.private.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request_id = state.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise InferenceExecutionError("execution has no submitted provider request")
    key = extract_unique_fal_key(key_source)
    import fal_client

    client = fal_client.SyncClient(key=key, default_timeout=120.0)
    status = client.status(LTX_V2V_APPLICATION, request_id, with_logs=False)
    status_name = type(status).__name__.lower()
    state["last_provider_status"] = status_name
    state["last_checked_at_utc"] = utc_now()
    atomic_write_json(state_path, state)
    if status_name != "completed":
        return {"phase": state["phase"], "provider_status": status_name}
    if getattr(status, "error", None):
        state["phase"] = "failed_pending_billing_verification"
        state["completed_at_utc"] = utc_now()
        state["provider_error_type"] = getattr(status, "error_type", None)
        atomic_write_json(state_path, state)
        update_budget_entry(
            budget_path,
            state["budget_label"],
            "failed_pending_billing_verification",
            completed_at_utc=state["completed_at_utc"],
        )
        return {"phase": state["phase"], "provider_status": status_name}
    try:
        result = client.result(LTX_V2V_APPLICATION, request_id)
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
        raise InferenceExecutionError("provider result has an unexpected shape")
    video_value = result.get("video")
    video_url = video_value.get("url") if isinstance(video_value, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith("https://"):
        raise InferenceExecutionError("completed result has no downloadable video")
    atomic_write_json(state_dir / "result.private.json", result)
    silent = state_dir / "generated_silent.mp4"
    exact_audio = state_dir / "generated_exact_audio.mp4"
    _download_video(video_url, silent)
    mux_exact_audio(silent, Path(state["audio_path"]), exact_audio)
    state.update(
        {
            "phase": "completed",
            "completed_at_utc": utc_now(),
            "generated_silent": {
                "size_bytes": silent.stat().st_size,
                "sha256": sha256_file(silent),
            },
            "generated_exact_audio": {
                "size_bytes": exact_audio.stat().st_size,
                "sha256": sha256_file(exact_audio),
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
        "outputs": [silent.name, exact_audio.name],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot latest-LTX V2V LoRA inference")
    subs = parser.add_subparsers(dest="command", required=True)
    start = subs.add_parser("start")
    start.add_argument("--video", type=Path, required=True)
    start.add_argument("--control", type=Path, required=True)
    start.add_argument("--lora", type=Path, required=True)
    start.add_argument("--audio", type=Path, required=True)
    start.add_argument("--expected-audio-sha256", required=True)
    start.add_argument("--anchor", type=Path)
    start.add_argument("--prompt", required=True)
    start.add_argument("--seed", type=int, required=True)
    start.add_argument("--lora-scale", type=float, default=1.0)
    start.add_argument("--num-frames", type=int, default=89)
    start.add_argument("--preserve-original-video", action="store_true")
    start.add_argument("--strength", type=float, default=1.0)
    start.add_argument("--video-strength", type=float, default=1.0)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--key-source", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--amount", type=float, default=0.13)
    monitor = subs.add_parser("monitor")
    monitor.add_argument("--budget", type=Path, required=True)
    monitor.add_argument("--key-source", type=Path, required=True)
    monitor.add_argument("--state-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "start":
        output = start_run(
            video=args.video,
            control=args.control,
            lora=args.lora,
            audio=args.audio,
            expected_audio_sha256=args.expected_audio_sha256,
            anchor=args.anchor,
            prompt=args.prompt,
            seed=args.seed,
            lora_scale=args.lora_scale,
            num_frames=args.num_frames,
            preserve_original_video=args.preserve_original_video,
            strength=args.strength,
            video_strength=args.video_strength,
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
