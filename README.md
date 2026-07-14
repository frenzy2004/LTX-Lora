# LTX Character LoRA Pilot

Reproducible tooling for a privacy-preserving, budget-capped LTX 2.3 character-LoRA pilot on fal.

## Safety boundaries

- Private source images, source videos, source audio, trained weights, provider URLs, and secrets stay outside Git. Only explicitly approved generated evaluation clips are published.
- All provider actions are dry-run unless `--execute` is supplied.
- Paid actions reserve projected cost in a local atomic ledger before submission.
- Paid jobs persist their provider request ID before log streaming so interrupted clients can recover results without resubmitting.
- The default pilot budget is **USD 12.00** and cannot be raised unless `ALLOW_BUDGET_OVERRIDE=1` is set explicitly.
- Dataset archives contain videos only; reference images are used only for validation and inference.
- The pilot uses a neutral trigger phrase and contains no personal names or chat material.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python scripts/inventory.py --source C:\path\to\private\videos --output private_work\inventory.json
python scripts/prepare_dataset.py `
  --inventory private_work\inventory.json `
  --selection private_work\selection.json `
  --output private_work\dataset
python scripts/train.py --dataset private_work\dataset\training.zip
```

The last command is a dry run. A paid request requires both a process-level `FAL_KEY` and `--execute`.

```powershell
$env:FAL_KEY = "set-outside-git"
python scripts/train.py --dataset private_work\dataset\training.zip --execute
```

## Pilot sequence

1. Inventory and validate private source media.
2. Select varied training clips and five held-out validation sources.
3. Center-crop clips to true 9:16, normalize to 720×1280 at 24 fps, and create caption sidecars.
4. Train a 500-step I2V LoRA smoke candidate.
5. Compare the adapter with the base model across the fixed location matrix.
6. Run exact-speech testing as a separate audio/lip-sync evaluation.
7. Publish sanitized quality, latency, and cost evidence.

## Budget-capped inference

The generation runner supports distilled text-to-video, image-to-video, base comparisons, and audio-driven LoRA tests. Uploaded private assets are cached by SHA-256 outside Git, provider request IDs are persisted before log streaming, and every paid request is reserved in the same USD 12 ledger.

```powershell
python scripts/generate.py `
  --name studio-smoke `
  --mode t2v-lora `
  --prompt "chrx9_person speaking to camera in a professional studio" `
  --lora-file private_work\adapter.safetensors
```

This is a dry run. Add `--execute` only after the projected request cost and remaining ledger budget have been reviewed.

See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for acceptance criteria.

## Current evidence

The first two fal generations are published under [`results/videos`](results/videos):

- `t2v-lora-prompt-only-identity-failure.mp4` documents that the 500-step adapter is not sufficient for prompt-only identity generation.
- `i2v-lora-reference-conditioned.mp4` shows substantially stronger identity preservation when the same adapter is combined with an approved first-frame reference.
- `a2v-lora-supplied-audio.mp4` tests an 11-second approved voice recording with the same first frame and adapter.

See [`results/EVALUATION.md`](results/EVALUATION.md) for the sanitized test record and current cost ledger.

## Manual selection file

Visual review should happen before dataset rendering. The optional private selection file pins reviewed source IDs and prevents automatic resolution-based selection from choosing rear views, hidden faces, or duplicate scenes:

```json
{
  "training_source_ids": ["source-id-a", "source-id-b"],
  "holdout_source_ids": ["source-id-c"]
}
```

The selection file remains private because its IDs map back to local source media.
