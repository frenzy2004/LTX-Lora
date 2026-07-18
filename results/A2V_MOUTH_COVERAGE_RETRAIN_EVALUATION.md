# LTX A2V Mouth-Coverage Retraining Evaluation

**Run date:** 2026-07-18
**Status:** Evidence collected; not cleared for an “indistinguishable from real footage” claim.

## What was tested

This was a controlled, Fal-hosted LTX-2.3 audio-to-video LoRA retrain and held-out evaluation. The purpose was to re-test speaking footage after correcting the earlier split so that the training partition included clips with visible inner-mouth speech motion.

| Item | Recorded setting |
|---|---|
| Training / holdout groups | 14 / 3 |
| Groups with visible inner-mouth speech motion | 2 training / 1 held-out |
| Training mode | LTX A2V LoRA |
| Training run | 400 steps, high 9:16, 89 frames at 24 fps |
| Evaluation model | LTX-2.3 Full 22B A2V LoRA |
| Evaluation scope | One held-out speaking clip, identical audio/start-image conditioning across all candidates |
| Output format | 576×960, 24 fps, 3.708 seconds, H.264 video + mono AAC audio |

The evaluation deliberately has only three generated candidates. It does **not** contain a new training run, an unbounded reroll loop, a non-LTX lip-sync product, raw source footage, private adapter weights, provider request IDs, or provider URLs.

## Generated outputs

All three files are generated LTX output and are safe to review in the repository.

| Candidate | LoRA stack | Generated video |
|---|---|---|
| A2V only | Retrained A2V adapter at 0.50 | [MP4](videos/a2v-mouth-retrain-new-a2v-only.mp4) |
| A2V + I2V (0.25) | Retrained A2V at 0.50; existing I2V adapter at 0.25 | [MP4](videos/a2v-mouth-retrain-new-a2v-i2v-025.mp4) |
| A2V + I2V (0.50) | Retrained A2V at 0.50; existing I2V adapter at 0.50 | [MP4](videos/a2v-mouth-retrain-new-a2v-i2v-050.mp4) |

Checksums and technical metadata are in [the public manifest](videos/a2v-mouth-retrain-evaluation.manifest.json).

## Evidence-based assessment

The files open successfully and include an audio stream. Frame-by-frame comparison against the withheld real clip shows that the generated candidates retain the same subject identity cues, garment, framing, and scene. They are visibly generated sequences rather than copied source video: at matched timestamps, head pose, gaze, and mouth geometry differ from the withheld footage.

That is useful evidence of identity-conditioned A2V generation, but it is **not** enough evidence to say the result is indistinguishable from genuine footage. In particular:

- This is a single held-out scene; it does not establish performance in other locations, lighting, cameras, or scripts.
- Static-frame review cannot certify natural temporal motion or exact audio-to-mouth synchronization for all frames.
- No independent, blinded “real versus generated” review has been conducted.

Therefore, this run is correctly described as a bounded technical evaluation, not a production approval or a claim that viewers would be unable to detect AI generation. The generated videos should be reviewed at normal playback speed by independent viewers before any external use.

## Cost ledger

| Ledger component | Amount (USD) |
|---|---:|
| Accounted spend before this retrain | 11.1209 |
| 400-step A2V retrain | 2.4000 |
| Three fixed Full 22B evaluations | 0.3000 |
| **Total accounted** | **13.8209** |
| Authorized ceiling | 14.0000 |
| Remaining authorization | 0.1791 |

No additional provider job was submitted outside this retrain and these three evaluation candidates.

## Reproducibility and privacy boundary

The public artifacts contain only the three generated MP4s, their hashes, aggregate configuration facts, and this assessment. They intentionally exclude raw training and hold-out footage, audio, images, training archives, weights, private configuration, provider URLs, request IDs, credentials, and participant identifiers.
