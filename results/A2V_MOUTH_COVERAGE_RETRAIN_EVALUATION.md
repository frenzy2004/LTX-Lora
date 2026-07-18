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

## Existing multi-location evidence

The repository also contains a separate, earlier Full-22B LTX A2V location matrix. It uses a different evaluation run, not the retrained adapter above, and provides three additional held-out speaking-location candidates:

| Held-out location | Generated video |
|---|---|
| Location 1 | [MP4](videos/a2v-full22b-lora-location-1.mp4) |
| Location 2 | [MP4](videos/a2v-full22b-lora-location-2.mp4) |
| Location 3 | [MP4](videos/a2v-full22b-lora-location-3.mp4) |

See [the location-matrix report](A2V_FULL22B_LOCATION_MATRIX.md) for its fixed configuration, media checks, and limitations. These clips expand the evidence across three held-out locations, but they do not replace a blinded real-versus-generated review and do not make an indistinguishability claim.

## Evidence-based assessment

The files open successfully and include an audio stream. Frame-by-frame comparison against the withheld real clip shows that the generated candidates retain the same subject identity cues, garment, framing, and scene. They are visibly generated sequences rather than copied source video: at matched timestamps, head pose, gaze, and mouth geometry differ from the withheld footage.

That is useful evidence of identity-conditioned A2V generation, but it is **not** enough evidence to say the result is indistinguishable from genuine footage. In particular:

- This is a single held-out scene; it does not establish performance in other locations, lighting, cameras, or scripts.
- Static-frame review cannot certify natural temporal motion or exact audio-to-mouth synchronization for all frames.
- No independent, blinded “real versus generated” review has been conducted.

Therefore, this run is correctly described as a bounded technical evaluation, not a production approval or a claim that viewers would be unable to detect AI generation. The generated videos should be reviewed at normal playback speed by independent viewers before any external use.

## Post-run visual and dataset failure review

**Conclusion: reject the current evidence package for the required realism standard.**

After the run, four evenly spaced frames were reviewed from every generated A2V MP4 currently published in this repository (14 files in total). This was a review of the complete public output set, not a selected highlight reel. It confirms the previously documented failures and adds the following bounded observations:

- [The managed trainer preview](videos/a2v-managed-trainer-preview.mp4) remains visibly inverted/negative-looking. It is not a usable talking-head result.
- [The managed prompt-only control](videos/a2v-managed-pure-lora-control.mp4) shows a different person. It is a direct identity failure.
- The reference-anchored and location-matrix outputs preserve more subject cues in still frames, but their public files do not establish natural temporal motion, reliable mouth-to-audio behavior, or indistinguishability at normal playback.
- In the three mouth-coverage retrain candidates above, matched-time stills differ from the withheld real footage in gaze/eyelid state, head pose, mouth shape, and fine surface detail. That proves the clips are generated rather than copied; it does **not** prove they look like genuine footage.

The standard requested for this project is stronger than “recognisable identity in a short generated clip.” It requires viewers to be unable to reliably distinguish generated material from real material. The current package does not meet that standard, and no public file in this repository should be described as having met it.

### What the execution data actually shows

The original selected source assets were all native 3840×2160 video. The actual submitted A2V set, and Fal's returned decoded-training archive, were different:

| Verified item | Observed value | Why it matters |
|---|---:|---|
| Selected original source clips | 17, all 3840×2160 | High-resolution originals existed. |
| Submitted A2V clips | 14 training / 3 hold-out | This was the complete retrain set. |
| Submitted and Fal-decoded video shape | 544×960, 24 fps, 89 frames | The run learned from short, portrait derivatives rather than native 4K frames. |
| Submitted clip duration | 3.708 seconds each | The run did not use the original source durations directly. |
| Explicit `[SPEECH]` caption tags | 0 of 17 | All captions contained natural-language speech descriptions, but none used the explicit A/V tag pattern recommended in the LTX guide. |
| Clips tagged during dataset review with visible inner-mouth speech motion | 2 training / 1 hold-out | This is limited coverage for evaluating a mouth-focused objective. |
| Cross-split isolation | One source asset and one location occur in both partitions; source intervals do not overlap | The hold-out is temporally disjoint, but it is not completely source-asset- or location-disjoint. |
| Retrain configuration | 400 steps, rank 32, learning rate 0.0002 | This differs from the official first-run starting point of roughly 1,000 steps at 1.5e-4. It is a difference to investigate, not proof of a single cause. |

The above facts do **not** prove that any one setting caused the poor visual result. They do establish that the previous test was not a native-4K, fully source-isolated, explicitly tagged speech-training evaluation. The official LTX guide says to debug in the order data → validation samples → inference setup → LoRA strength → learning rate/steps/rank; this audit follows that order. [LTX training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

### Local-machine constraint

This PC had 26.38 GiB free during the audit. That is sufficient for the existing private dataset, Fal-managed preparation, results, and review artifacts, but it is not sufficient for a local LTX-2.3 22B checkpoint plus its supporting models and working files. The official local trainer also requires Linux/CUDA and recommends an 80 GB GPU for the standard configuration. [LTX Quick Start](https://docs.ltx.io/open-source-model/ltx-trainer/quick-start)

Consequently, a local full-model retrain is not a viable remediation on this machine. Fal-managed LTX training remains technically possible; it must be treated as a new, controlled evaluation rather than a continuation of the rejected run.

### Required gate before another paid run

1. Build a new, private source-to-training manifest from the highest-quality usable source regions and explicitly document the portrait crop/resize policy.
2. Use Fal's `debug_dataset` output to inspect decoded videos, audio, captions, duration, and frame rate **before** submitting a full training job. Fal documents this field specifically for pre-commit dataset inspection. [Fal A2V Trainer API](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api)
3. Keep the current natural-language captions, but add an explicit A/V speech tag and verify it in the private archive; the LTX guide recommends explicit A/V tagging or an equally explicit speech description. [LTX training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)
4. Define a held-out set with source-asset and location separation where the available material permits it, then keep inference settings fixed for comparison.
5. Run a blinded, randomized real-versus-generated playback test before making any claim about indistinguishability. LTX's own guidance proposes comparing roughly ten real shots and ten generated shots with roughly ten reviewers. [LTX training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

No new provider request was submitted while preparing this failure review.

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
