# Broad A2V Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a deterministic, face-aware, source-isolated Fal LTX-2.3 A2V training archive from a broad real-video corpus.

**Architecture:** A reusable Git-safe Python tool owns pure split/window/crop logic, FFmpeg rendering, validation, review-sheet metadata, and provider-mirror construction. Private source paths, word timestamps, captions, trigger tokens, and generated media are supplied only at runtime and remain in the private run workspace.

**Tech Stack:** Python 3.12, standard library, OpenCV YuNet, FFmpeg/FFprobe, unittest, existing `ltx_lora_pilot.a2v_dataset` structural validator.

## Global Constraints

- Canonical target: 544×960, 89 frames, 24 fps, silent H.264 target plus mono 48 kHz PCM WAV. Face-aware crops use the exact 17:30 bucket aspect (approximately 9:16) to prevent anisotropic face stretching during resize.
- Train/holdout separation is by source file; seed 42; roughly 10 percent holdout with minimum five when the passing source count permits.
- The normal-colour local dataset is authoritative; a provider-only pre-inverted mirror exists solely to counter the previously measured Fal channel inversion.
- A 100-step provider-decoding gate must pass before one 4,000-step, rank-32, learning-rate-0.0001 request.
- One paid request in flight maximum; no ambiguous resubmission.
- Derived private workspace ceiling: 8 GiB.
- Never commit private paths, aliases tied to filenames, raw media, audio, captions/transcripts, trigger tokens, credentials, request IDs, signed URLs, LoRA weights, or provider artifact URLs.
- Preserve unrelated dirty worktree changes and stage explicit pathspecs only.

---

### Task 1: Deterministic source split and speech-window selection

**Files:**
- Create: `tools/a2v_broad_dataset.py`
- Create: `tests/test_a2v_broad_dataset.py`

**Interfaces:**
- Produces: `split_sources(source_ids: Sequence[str], holdout_fraction: float, min_holdout: int, seed: int) -> tuple[list[str], list[str]]`
- Produces: `select_speech_windows(words: Sequence[Word], source_duration: float, clip_seconds: float, max_windows: int) -> list[Window]`

- [ ] **Step 1: Write failing tests for source isolation and deterministic windows**

```python
def test_split_sources_is_deterministic_and_disjoint():
    train, holdout = split_sources([f"src-{i:03d}" for i in range(50)], 0.10, 5, 42)
    assert len(holdout) == 5
    assert set(train).isdisjoint(holdout)
    assert (train, holdout) == split_sources([f"src-{i:03d}" for i in range(50)], 0.10, 5, 42)

def test_select_speech_windows_returns_exact_non_overlapping_intervals():
    words = [Word(start=i * 0.45, end=i * 0.45 + 0.30, text=f"w{i}") for i in range(30)]
    windows = select_speech_windows(words, 20.0, 89 / 24, 2)
    assert len(windows) == 2
    assert all(abs(window.duration - 89 / 24) < 1e-9 for window in windows)
    assert windows[0].end <= windows[1].start or windows[1].end <= windows[0].start
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected: import failure because `tools.a2v_broad_dataset` does not exist.

- [ ] **Step 3: Implement minimal immutable records and pure selection functions**

```python
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

def split_sources(source_ids, holdout_fraction=0.10, min_holdout=5, seed=42):
    ordered = sorted(set(source_ids))
    count = min(len(ordered) - 1, max(min_holdout, round(len(ordered) * holdout_fraction)))
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    holdout = sorted(shuffled[:count])
    return sorted(set(ordered) - set(holdout)), holdout
```

Implement `select_speech_windows` by scoring exact-duration candidate intervals at word starts, preferring higher word count and speech coverage, then greedily retaining non-overlapping windows.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected: both Task 1 tests pass.

---

### Task 2: Stable face-aware portrait crop

**Files:**
- Modify: `tools/a2v_broad_dataset.py`
- Modify: `tests/test_a2v_broad_dataset.py`

**Interfaces:**
- Produces: `derive_portrait_crop(frame_size: tuple[int, int], observations: Sequence[FaceObservation], output_aspect: float = 9 / 16) -> CropDecision`

- [ ] **Step 1: Write failing crop tests**

```python
def test_crop_tracks_off_centre_face_and_contains_temporal_envelope():
    observations = [
        FaceObservation(0.0, primary=Box(2800, 500, 420, 500), secondary=None),
        FaceObservation(1.0, primary=Box(2920, 520, 430, 500), secondary=None),
        FaceObservation(2.0, primary=Box(2850, 510, 425, 510), secondary=None),
    ]
    decision = derive_portrait_crop((3840, 2160), observations)
    assert decision.accepted
    assert abs(decision.crop.width / decision.crop.height - 9 / 16) < 0.002
    assert decision.crop.x > (3840 - decision.crop.width) / 2
    assert all(decision.crop.contains(item.primary) for item in observations)

def test_crop_rejects_prominent_second_face():
    observations = [FaceObservation(0.0, Box(1000, 400, 500, 600), Box(1800, 420, 470, 580))]
    assert derive_portrait_crop((3840, 2160), observations).reason == "prominent_second_face"
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `python -m unittest tests.test_a2v_broad_dataset.BroadDatasetTests.test_crop_tracks_off_centre_face_and_contains_temporal_envelope tests.test_a2v_broad_dataset.BroadDatasetTests.test_crop_rejects_prominent_second_face -v`

Expected: missing face/crop types.

- [ ] **Step 3: Implement the minimum crop model**

Use the median primary-face size, the union of primary boxes, a face-height target near 28 percent of the crop, and a downward composition bias that keeps the face near the upper third. Expand to contain the complete temporal face envelope plus hair/jaw margin, clamp to the source frame, round coordinates to even integers, and reject insufficient detection coverage or a secondary/primary area ratio above the configured threshold.

- [ ] **Step 4: Run all tests and verify GREEN**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected: Task 1 and Task 2 tests pass.

---

### Task 3: Exact A2V rendering and validation

**Files:**
- Modify: `tools/a2v_broad_dataset.py`
- Modify: `tests/test_a2v_broad_dataset.py`

**Interfaces:**
- Produces: `render_group(source: Path, window: Window, crop: Crop, destination: Path, basename: str, caption: str) -> GroupPaths`
- Produces: `validate_group(paths: GroupPaths) -> GroupAudit`

- [ ] **Step 1: Write a failing FFmpeg integration test**

Create a five-second 1920×1080/60 fps test video with a 48 kHz tone, render one group, and assert:

```python
assert audit.width == 544 and audit.height == 960
assert audit.frames == 89 and audit.fps == 24
assert audit.audio_rate == 48_000 and audit.audio_channels == 1
assert audit.start_matches_target_first_frame
assert not audit.target_has_audio
```

- [ ] **Step 2: Run the integration test and verify RED**

Expected: `render_group` is missing.

- [ ] **Step 3: Implement deterministic FFmpeg rendering**

Render the target with `crop`, Lanczos `scale=544:960`, `fps=24`, `-frames:v 89`, H.264, yuv420p, no audio, and faststart. Extract 48 kHz mono PCM WAV from the identical source start/duration. Decode the target's first frame to RGB PNG after rendering. Write the approved caption exactly once.

- [ ] **Step 4: Implement validation with FFprobe and decoded-frame hashing**

Require exact stream properties, compare RGB pixels from `_start.png` with frame zero decoded from `_end.mp4`, and fail closed on any mismatch.

- [ ] **Step 5: Run all tests and verify GREEN**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected: all tests pass with FFmpeg available.

---

### Task 4: Provider-only colour mirror and archive integrity

**Files:**
- Modify: `tools/a2v_broad_dataset.py`
- Modify: `tests/test_a2v_broad_dataset.py`

**Interfaces:**
- Produces: `build_provider_mirror(canonical_dir: Path, mirror_dir: Path) -> MirrorAudit`
- Produces: `write_training_archive(groups_dir: Path, archive: Path) -> ArchiveAudit`

- [ ] **Step 1: Write failing mirror tests**

```python
def test_provider_mirror_inverts_only_visual_pixels(tmp_path):
    canonical = make_fixture_group(tmp_path / "canonical")
    audit = build_provider_mirror(canonical, tmp_path / "mirror")
    assert audit.audio_sha256_equal
    assert audit.caption_sha256_equal
    assert audit.visual_inverse_mean_absolute_error < 5.0
```

- [ ] **Step 2: Run the mirror test and verify RED**

Expected: `build_provider_mirror` is missing.

- [ ] **Step 3: Implement mirror and archive checks**

Invert RGB video and start-image pixels only. Copy WAV and caption byte-for-byte. Verify the inverse relationship by decoding pixels, then create a ZIP containing exactly four files for every training basename and no holdout files.

- [ ] **Step 4: Run all tests and verify GREEN**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected: all tests pass.

---

### Task 5: Private corpus orchestration and complete visual audit

**Files:**
- Modify: `tools/a2v_broad_dataset.py`
- Create privately at runtime: `projects/a2v-participant-broad-4000-20260719-2146/dataset/source-audit.private.json`
- Create privately at runtime: `projects/a2v-participant-broad-4000-20260719-2146/dataset/dataset.json`
- Create privately at runtime: `projects/a2v-participant-broad-4000-20260719-2146/dataset/holdout.jsonl`

**Interfaces:**
- CLI consumes private source root, source inventory, Whisper word-timestamp directory, approved caption mapping, YuNet model path, output root, and 8 GiB ceiling.
- CLI produces canonical training/holdout groups, provider mirror/archive, private manifests, exclusion reasons, and review contact sheets.

- [ ] **Step 1: Write failing CLI dry-run tests**

Assert that `--dry-run` writes only selection/audit metadata, that holdout source IDs never appear in training, and that a projected size over 8 GiB exits nonzero before rendering.

- [ ] **Step 2: Verify RED, implement CLI, and verify GREEN**

Run: `python -m unittest tests.test_a2v_broad_dataset -v`

Expected after implementation: all tests pass.

- [ ] **Step 3: Transcribe eligible sources locally and discover candidate windows**

Use cached `faster-whisper-small.en` on CPU/int8 with word timestamps. Persist private transcripts under the run workspace only. Do not put transcript text in captions or Git.

- [ ] **Step 4: Run face analysis and render low-cost review proxies first**

Inspect every selected window's full contact strip with face/crop overlays. Reject obstruction, mouth invisibility, cuts, prominent second faces, filters, defocus, or unstable framing before full-quality rendering.

- [ ] **Step 5: Render canonical groups, validate, and build the provider mirror/archive**

Print passing source/group/holdout counts and aggregate exclusion reasons. Do not claim a predetermined group count.

---

### Task 6: Provider decoding gate, 4,000-step run, and evaluation

**Files:**
- Create: `tools/run_a2v_broad_provider.py`
- Create: `tests/test_run_a2v_broad_provider.py`
- Create privately at runtime: `projects/a2v-participant-broad-4000-20260719-2146/outputs/provider-debug-100/`
- Create privately at runtime: `projects/a2v-participant-broad-4000-20260719-2146/outputs/provider-main-4000/`
- Create Git-safe: `results/BROAD_A2V_4000_EVALUATION.md`

- [ ] **Step 1: Refresh official Fal schema/price evidence and reserve $0.60**

Stop if the displayed rate is not $0.006/step or if another paid request is in flight.

- [ ] **Step 2: Submit exactly one 100-step request with logs and debug dataset enabled**

Persist the request identity privately before polling. Never resubmit an ambiguous request.

- [ ] **Step 3: Download and compare the decoded archive**

Require normal-colour closeness to canonical masters, exact group count, 544×960/89/24 video, 48 kHz audio, and intact captions. Stop on mismatch.

- [ ] **Step 4: Reserve $24 and submit exactly one 4,000-step request**

Poll with provider logs enabled. Persist every emitted progress, validation, warning, and loss value; explicitly record when loss is not exposed.

- [ ] **Step 5: Evaluate source-isolated holdouts and the exact 10.857333-second audio**

Compare the 1,000-step and 4,000-step adapters on identical settings, then run the primary exact-audio request without an image and one controlled held-out real-frame diagnostic. Review every frame and full audio before presenting results.

- [ ] **Step 6: Publish only sanitized methodology and honest results**

Run secret/private-path scans, stage explicit new paths only, commit, push the existing feature branch, and report commit hashes.
