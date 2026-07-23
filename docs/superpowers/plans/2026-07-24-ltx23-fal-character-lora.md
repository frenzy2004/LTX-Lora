# LTX-2.3 FAL Character LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a budget-capped LTX-2.3/FAL character LoRA pipeline that prepares tutorial-style clips, trains a LoRA, and saves generated video outputs.

**Architecture:** Add focused helpers to the existing `ltx_lora_pilot` package for LTX-2.3 V2 training and quality LoRA inference, plus a dataset builder that preserves audio and writes neutral `orvo` captions. Use existing FAL upload/submit and budget ledger patterns, then run private media work under ignored `private_work/` and publish only final clips under the workspace `outputs/`.

**Tech Stack:** Python 3.9+, pytest, FFmpeg via `imageio-ffmpeg`, local Whisper CLI, `fal-client`, FAL `fal-ai/ltx23-trainer-v2/t2v`, FAL `fal-ai/ltx-2.3-quality/*/lora`.

## Global Constraints

- Use only LTX models and LTX/FAL endpoints.
- Use the current LTX-2.3 trainer and inference endpoints.
- Keep a hard local budget cap of USD 25.00.
- Do not use the user's real name in trigger phrases, captions, prompts, filenames, model labels, or tests.
- Use an invented neutral trigger, `orvo`, in captions and render prompts.
- Treat API keys as secrets. Read them from process environment only and do not write them into repo files.
- Save final user-facing videos and manifests under the workspace `outputs/` directory.

---

## File Structure

- Create `src/ltx_lora_pilot/ltx23_v2.py`: endpoint constants, cost estimators, training payload builder, quality inference payload builder.
- Create `scripts/prepare_ltx23_tutorial_dataset.py`: transcribe source audio, choose clean transcript windows, render short audio-preserving 9:16 clips, write `.txt` sidecars, and zip the training directory.
- Create `scripts/run_ltx23_fal_pipeline.py`: upload dataset, submit training, download LoRA/config, submit 2-3 LTX-2.3 quality renders, download videos, and write final manifest.
- Create `tests/test_ltx23_v2.py`: verify endpoints, budget math, payload schemas, neutral trigger privacy, and frame/aspect validation.
- Create `tests/test_prepare_ltx23_dataset.py`: verify caption sanitization, clip selection from transcript segments, and zip file layout.
- Modify only if needed: `src/ltx_lora_pilot/budget.py` to support explicit USD 25.00 cap without weakening lower default safety.

---

### Task 1: LTX-2.3 V2 Payload Helpers

**Files:**
- Create: `src/ltx_lora_pilot/ltx23_v2.py`
- Test: `tests/test_ltx23_v2.py`

**Interfaces:**
- Produces: `estimate_ltx23_t2v_training_cost(steps: int) -> Decimal`
- Produces: `estimate_ltx23_quality_inference_cost(seconds: Decimal | float | int, resolution: str) -> Decimal`
- Produces: `build_ltx23_t2v_training_payload(**kwargs) -> dict[str, object]`
- Produces: `build_ltx23_quality_lora_payload(mode: str, prompt: str, lora_url: str, image_url: str | None = None, seed: int | None = None) -> tuple[str, dict[str, object]]`

- [ ] **Step 1: Write failing tests**

```python
from decimal import Decimal

import pytest

from ltx_lora_pilot.ltx23_v2 import (
    LTX23_T2V_TRAINER_ENDPOINT,
    estimate_ltx23_quality_inference_cost,
    estimate_ltx23_t2v_training_cost,
    build_ltx23_quality_lora_payload,
    build_ltx23_t2v_training_payload,
)


def test_training_cost_uses_current_fal_v2_t2v_rate() -> None:
    assert estimate_ltx23_t2v_training_cost(2000) == Decimal("12.0000")


def test_training_payload_uses_private_neutral_trigger() -> None:
    payload = build_ltx23_t2v_training_payload(training_data_url="https://private.invalid/training.zip")
    assert payload["trigger_phrase"] == "orvo"
    assert payload["number_of_steps"] == 2000
    assert payload["number_of_frames"] == 121
    assert payload["frame_rate"] == 24
    assert payload["aspect_ratio"] == "9:16"
    assert payload["with_audio"] is True
    assert "realname" not in str(payload).lower()
    assert "surname" not in str(payload).lower()


def test_quality_t2v_lora_payload_matches_fal_schema() -> None:
    endpoint, payload = build_ltx23_quality_lora_payload(
        mode="t2v",
        prompt='orvo says, "Coffee mugs have stronger opinions than most meetings."',
        lora_url="https://private.invalid/orvo.safetensors",
        seed=7,
    )
    assert endpoint == "fal-ai/ltx-2.3-quality/text-to-video/lora"
    assert payload["resolution"] == "portrait_16_9"
    assert payload["num_frames"] == 121
    assert payload["frames_per_second"] == 24
    assert payload["generate_audio"] is True
    assert payload["loras"] == [
        {"path": "https://private.invalid/orvo.safetensors", "scale": 1.0, "transformer": "both"}
    ]


def test_quality_i2v_requires_image_url() -> None:
    with pytest.raises(ValueError, match="image_url"):
        build_ltx23_quality_lora_payload(
            mode="i2v",
            prompt='orvo says, "This is a calm test."',
            lora_url="https://private.invalid/orvo.safetensors",
        )


def test_quality_inference_cost_is_second_based() -> None:
    assert estimate_ltx23_quality_inference_cost(5, "1080p") == Decimal("0.3000")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/test_ltx23_v2.py -v`
Expected: fail with `ModuleNotFoundError: No module named 'ltx_lora_pilot.ltx23_v2'`.

- [ ] **Step 3: Implement helpers**

Create `src/ltx_lora_pilot/ltx23_v2.py` with endpoint constants, rate constants, validation, and the payload builders specified in the tests.

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=src pytest tests/test_ltx23_v2.py -v`
Expected: 5 passed.

---

### Task 2: Tutorial-Style Dataset Builder

**Files:**
- Create: `scripts/prepare_ltx23_tutorial_dataset.py`
- Test: `tests/test_prepare_ltx23_dataset.py`

**Interfaces:**
- Consumes: `orvo` trigger from `ltx_lora_pilot.ltx23_v2.TRIGGER`
- Produces: `sanitize_caption(text: str) -> str`
- Produces: `choose_clip_windows(segments: list[dict[str, object]], target_count: int, min_seconds: float, max_seconds: float) -> list[dict[str, object]]`
- Produces: `write_training_zip(source_video: Path, transcript_json: Path, output_dir: Path, ffmpeg: str, target_count: int) -> Path`

- [ ] **Step 1: Write failing tests**

```python
import json
import zipfile
from pathlib import Path

from ltx_lora_pilot.ltx23_v2 import TRIGGER
from scripts.prepare_ltx23_tutorial_dataset import choose_clip_windows, sanitize_caption


def test_sanitize_caption_removes_blocked_names_and_uses_trigger() -> None:
    caption = sanitize_caption('Realname Surname says hello and talks about coffee.')
    assert caption.startswith(f"{TRIGGER} says, ")
    assert "realname" not in caption.lower()
    assert "surname" not in caption.lower()
    assert "coffee" in caption


def test_choose_clip_windows_prefers_short_sentence_spans() -> None:
    segments = [
        {"start": 0.0, "end": 1.2, "text": "Too short."},
        {"start": 2.0, "end": 6.5, "text": "This is a useful sentence about a notebook."},
        {"start": 8.0, "end": 15.5, "text": "This useful line talks about a sandwich and a meeting."},
        {"start": 20.0, "end": 31.0, "text": "Too long for the tutorial style."},
    ]
    windows = choose_clip_windows(segments, target_count=2, min_seconds=2.0, max_seconds=8.0)
    assert [round(window["duration"], 1) for window in windows] == [4.5, 7.5]
    assert all(window["caption"].startswith(f"{TRIGGER} says, ") for window in windows)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src:. pytest tests/test_prepare_ltx23_dataset.py -v`
Expected: fail because the script does not exist.

- [ ] **Step 3: Implement dataset builder**

Create the script with pure helper functions plus a CLI. The CLI accepts `--source-video`, `--transcript-json`, `--output-dir`, `--ffmpeg`, and `--target-count`. FFmpeg command must preserve audio with `-map 0:v:0 -map 0:a:0?`, normalize display orientation, crop/scale to `720:1280`, encode `libx264`/AAC, write captions next to clips, and create `training.zip`.

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=src:. pytest tests/test_prepare_ltx23_dataset.py -v`
Expected: 2 passed.

---

### Task 3: Budget-Capped FAL Runner

**Files:**
- Create: `scripts/run_ltx23_fal_pipeline.py`
- Test: `tests/test_ltx23_v2.py`

**Interfaces:**
- Consumes: payload/cost helpers from `ltx_lora_pilot.ltx23_v2`
- Consumes: `BudgetLedger` from `ltx_lora_pilot.budget`
- Produces: `private_work/ltx23_orvo/training_result.json`
- Produces: `private_work/ltx23_orvo/lora.safetensors`
- Produces: `../outputs/ltx23_orvo_manifest.json`
- Produces: `../outputs/ltx23_orvo_*.mp4`

- [ ] **Step 1: Implement runner**

Create a runner that accepts `--dataset-zip`, `--budget-state`, `--private-output-dir`, `--public-output-dir`, `--execute`, and reads `FAL_KEY` from the environment. It uploads the dataset, submits `fal-ai/ltx23-trainer-v2/t2v`, saves request IDs before waiting, downloads returned `lora_file` and `config_file`, then submits at least two `fal-ai/ltx-2.3-quality/text-to-video/lora` prompts with random-topic speech.

- [ ] **Step 2: Dry-run runner**

Run: `PYTHONPATH=src python scripts/run_ltx23_fal_pipeline.py --dataset-zip private_work/ltx23_orvo/dataset/training.zip --budget 25 --dry-run`
Expected: print projected training cost, planned prompts, and no provider submission.

- [ ] **Step 3: Execute with hidden FAL key**

Run inside a TTY with echo disabled, read `FAL_KEY`, then call the runner with `--execute`.
Expected: request IDs are saved before blocking waits, budget ledger never exceeds USD 25.00.

---

### Task 4: Private Data Prep and Final Verification

**Files:**
- Create ignored private files under `private_work/ltx23_orvo/`
- Create final files under workspace `outputs/`

**Interfaces:**
- Consumes: `scripts/prepare_ltx23_tutorial_dataset.py`
- Consumes: `scripts/run_ltx23_fal_pipeline.py`

- [ ] **Step 1: Transcribe source**

Run local Whisper on `IMG_3816-001.MOV` extracted audio and save JSON under `private_work/ltx23_orvo/source_transcript.json`.

- [ ] **Step 2: Build dataset**

Run `scripts/prepare_ltx23_tutorial_dataset.py` for 20 clips.
Expected: `private_work/ltx23_orvo/dataset/training.zip` with 20 `.mp4` and 20 `.txt` files.

- [ ] **Step 3: Privacy and media verification**

Run: `rg -i "<blocked-name-pattern>" private_work/ltx23_orvo/dataset/training`
Expected: no matches.
Run FFmpeg probe on sample clips.
Expected: each clip has one video stream and one audio stream.

- [ ] **Step 4: Execute FAL pipeline**

Run the paid runner after passing the FAL key through the environment.
Expected: LoRA weights/config and final `.mp4` outputs downloaded locally.

- [ ] **Step 5: Verify final outputs**

Use FFmpeg to inspect final videos for portrait resolution, audio stream, and nonzero duration. Create or update the final manifest with local paths, seeds, prompt text, FAL request IDs, and budget ledger summary.
