# LTX Character LoRA Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate one private LTX 2.3 character LoRA through fal while keeping public tooling reproducible and total provider spend at or below USD 12.

**Architecture:** Public code, configuration examples, tests, and sanitized aggregate results live in this repository. Private source media, generated datasets, provider credentials, adapter weights, signed URLs, and raw outputs remain in ignored local storage. Paid operations reserve estimated cost in an atomic local ledger before submission and require an explicit `--execute` flag.

**Tech Stack:** Python 3.11+, pytest, ffmpeg/ffprobe, fal-client, LTX 2.3 managed training and custom-LoRA inference.

## Global Constraints

- Do not commit private media, subject identifiers, credentials, signed URLs, adapter weights, or raw generated outputs.
- Enforce a USD 12 hard provider-spend cap; override is disabled unless `ALLOW_BUDGET_OVERRIDE=1` is explicitly set.
- Use a video-only training archive with sidecar captions and five held-out validation clips.
- Start with a 500-step I2V LoRA smoke run before considering longer training.
- Keep exact scripted speech evaluation separate from native model audio evaluation.
- Push only after tests, compilation, secret scanning, and repository-diff checks pass.

---

### Task 1: Public scaffold and privacy boundary

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `tests/test_privacy.py`
- Create: `docs/superpowers/plans/2026-07-14-ltx-lora-pilot.md`

**Interfaces:**
- Consumes: repository root and Git index.
- Produces: a public repository in which private file types and secret-like strings are rejected by tests.

- [ ] **Step 1: Reproduce the privacy-test failure**

Run: `python -m pytest tests/test_privacy.py -q`

Expected: FAIL because the test source contains the literal private terms it scans for.

- [ ] **Step 2: Replace identity-specific literals with structural checks**

The test must reject tracked media extensions, credential patterns, and private workspace directories without embedding any private identity terms.

- [ ] **Step 3: Verify the privacy test**

Run: `python -m pytest tests/test_privacy.py -q`

Expected: PASS.

- [ ] **Step 4: Commit the verified scaffold**

Run: `git add . && git commit -m "chore: scaffold private LTX LoRA pilot"`

Expected: one root commit with no private files or credentials.

### Task 2: Budget reservation and conservative failure accounting

**Files:**
- Modify: `src/ltx_lora_pilot/budget.py`
- Modify: `scripts/train.py`
- Modify: `tests/test_budget.py`

**Interfaces:**
- Consumes: `BudgetLedger.reserve(amount, description)` and `BudgetLedger.finalize(reservation_id, actual_amount)`.
- Produces: a paid-job path that never releases a reservation after provider submission may have occurred.

- [ ] **Step 1: Add a failing submission-state test**

Test that an exception raised after the submit boundary leaves the conservative estimated charge finalized instead of releasing it.

- [ ] **Step 2: Run the focused test**

Run: `python -m pytest tests/test_budget.py -q`

Expected: FAIL on the post-submit exception case.

- [ ] **Step 3: Move the submitted-state boundary before the remote submit call**

Set the conservative state before invoking the provider operation; release only failures that occur before that boundary.

- [ ] **Step 4: Verify all budget tests**

Run: `python -m pytest tests/test_budget.py -q`

Expected: PASS with reservation, cap, release, and post-submit cases covered.

### Task 3: Safe dataset preparation

**Files:**
- Modify: `scripts/prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: a private inventory JSON and a private output directory.
- Produces: `training/`, `holdout/`, `training_dataset.zip`, and a private selection manifest.

- [ ] **Step 1: Add failing cleanup-boundary tests**

Test that cleanup accepts only the explicit `training` and `holdout` children beneath the resolved output root and refuses any other path.

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_prepare_dataset.py -q`

Expected: FAIL because the boundary helper does not yet exist.

- [ ] **Step 3: Implement the cleanup guard**

Resolve both paths, require `candidate.parent == output_root`, and require `candidate.name in {"training", "holdout"}` before recursive removal.

- [ ] **Step 4: Verify the focused tests**

Run: `python -m pytest tests/test_prepare_dataset.py -q`

Expected: PASS.

### Task 4: Private inventory, visual QA, and selection

**Files:**
- Create outside Git: `../work/character_lora_pilot/private_work/inventory.json`
- Create outside Git: `../work/character_lora_pilot/private_work/contact_sheets/`
- Create outside Git: `../work/character_lora_pilot/private_work/selection.json`

**Interfaces:**
- Consumes: downloaded source videos and the private reference set.
- Produces: 30 diverse training clips plus five held-out clips with no overlap.

- [ ] **Step 1: Build the private inventory**

Run: `python scripts/inventory.py --source ../work/source_media --output ../work/character_lora_pilot/private_work/inventory.json`

Expected: JSON with probe-success counts, duration, dimensions, frame rate, and source-only private paths.

- [ ] **Step 2: Generate representative thumbnails**

Use ffmpeg to extract midpoint frames for candidates into the ignored contact-sheet directory.

- [ ] **Step 3: Review diversity and exclusions**

Reject corrupt, tiny, blurred, duplicate, multi-person, watermarked, or visually obstructed clips. Balance frontal, three-quarter, profile, close, medium, expression, clothing, lighting, and background coverage.

- [ ] **Step 4: Prepare and validate the archive**

Run: `python scripts/prepare_dataset.py --inventory ../work/character_lora_pilot/private_work/inventory.json --output ../work/character_lora_pilot/private_work/dataset --train-count 30 --holdout-count 5`

Expected: exactly 30 training videos, 30 matching caption files, five holdouts outside the ZIP, and no audio stream in training clips.

### Task 5: Initial public push and dry-run audit

**Files:**
- Modify: `docs/TEST_PLAN.md`
- Create after measurements: `results/pilot-summary.json`

**Interfaces:**
- Consumes: tested repository and sanitized aggregate measurements.
- Produces: pushed public tooling and a zero-cost dry-run record.

- [ ] **Step 1: Run the complete local verification**

Run: `python -m pytest -q && python -m compileall -q src scripts && git diff --check`

Expected: all tests pass, compilation succeeds, and no whitespace errors appear.

- [ ] **Step 2: Scan tracked content**

Run: `git grep -n -I -E "(FAL_KEY=.+|[A-Za-z0-9_-]{24,}:[A-Za-z0-9_-]{24,})" -- . ':!.env.example'`

Expected: no matches.

- [ ] **Step 3: Push the public checkpoint**

Run: `git push -u origin main`

Expected: the remote main branch contains only public tooling and documentation.

- [ ] **Step 4: Run the training dry-run**

Run: `python scripts/train.py --steps 500 --dataset ../work/character_lora_pilot/private_work/dataset/training.zip`

Expected: projected training cost USD 1.20, no provider job, and zero ledger consumption.

### Task 6: Capped fal smoke training

**Files:**
- Create outside Git: `../work/character_lora_pilot/private_work/provider/`
- Modify after sanitization: `results/pilot-summary.json`

**Interfaces:**
- Consumes: `FAL_KEY` from process environment, verified archive, USD 12 ledger.
- Produces: one private adapter artifact and sanitized job timing/cost metadata.

- [ ] **Step 1: Validate authentication without printing the credential**

Load the credential into the current process environment and confirm only that it is non-empty; never print its value.

- [ ] **Step 2: Submit one 500-step I2V training job**

Run: `python scripts/train.py --steps 500 --dataset ../work/character_lora_pilot/private_work/dataset/training.zip --execute`

Expected: estimated reservation USD 1.20, provider job completion, private adapter output, and ledger consumption no greater than USD 1.20 unless provider-reported usage proves otherwise.

- [ ] **Step 3: Validate the private artifact**

Check that the adapter URL or downloaded weight is present, tied to LTX 2.3, and stored only outside Git.

- [ ] **Step 4: Push sanitized training metadata**

Record endpoint, steps, elapsed time, status, and cost without IDs, URLs, paths, hashes, or identity metadata; commit and push.

### Task 7: Location and speaking evaluation

**Files:**
- Create: `scripts/generate.py`
- Create: `tests/test_generation_budget.py`
- Modify: `results/pilot-summary.json`
- Create: `results/evaluation.md`

**Interfaces:**
- Consumes: private adapter, neutral test prompts, private first-frame references, private voice sample.
- Produces: base-versus-LoRA comparisons, native-audio observations, exact-speech pipeline observations, and accepted-output cost.

- [ ] **Step 1: Add failing generation-cost tests**

Cover duration, resolution, model tier, attempt count, remaining-budget refusal, and dry-run behavior.

- [ ] **Step 2: Implement the provider-neutral generation wrapper**

Require `--execute`, reserve maximum request cost, use only the selected private adapter, and store raw outputs outside Git.

- [ ] **Step 3: Generate the minimum informative matrix**

Test studio, office, podcast, stage, cafe, library, outdoor, and abstract-background prompts using short 720p clips. Run base and LoRA comparisons first; expand seeds only while remaining under the USD 12 cap.

- [ ] **Step 4: Evaluate speech honestly**

Test native audiovisual speech for plausibility and test the supplied recording through an exact-speech/lip-sync stage only if a supported route fits the remaining budget. Do not claim native LTX reproduces exact words unless verified.

- [ ] **Step 5: Publish the sanitized conclusion**

Report identity consistency, prompt adherence, motion, artifacts, latency, paid attempts, accepted outputs, actual spend, and whether the cost advantage is supported. Commit and push all public results.

### Task 8: Final verification and handoff

**Files:**
- Modify: `README.md`
- Modify: `results/evaluation.md`

**Interfaces:**
- Consumes: all tests, Git history, budget ledger, and evaluation evidence.
- Produces: a reproducible CTO-facing pilot record and a private artifact handoff.

- [ ] **Step 1: Run final verification**

Run: `python -m pytest -q && python -m compileall -q src scripts && git diff --check && git status --short`

Expected: tests pass, compilation succeeds, no diff errors, and only intended files are present.

- [ ] **Step 2: Confirm provider spend**

Read the private ledger and provider-reported costs; verify total actual or conservatively estimated spend is at most USD 12.

- [ ] **Step 3: Confirm remote state**

Run: `git push && git log -5 --oneline && git status --short`

Expected: remote is current and working tree is clean.

- [ ] **Step 4: Hand off artifacts**

Provide links to the public repository files and describe the private adapter/output locations without exposing credentials or signed URLs.
