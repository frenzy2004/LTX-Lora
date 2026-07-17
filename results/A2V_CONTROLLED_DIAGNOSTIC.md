# LTX A2V controlled diagnostic

## Decision

**Do not use this adapter for a production talking-head release.** The controlled outputs below still show visible synthetic cues. They do not meet the requirement that a typical social-media viewer should be unable to recognize the clip as generated.

This is a diagnosis of one specific managed A2V LoRA run. It is not a claim that every LTX workflow will have the same result.

## What this test controlled

All three clips use the same held-out real start frame, the same held-out speech audio, the same 544×960 / 89-frame / 24 fps output specification, the same prompt, and the same seed. The only variable is the model path:

| Control | Adapter scale | Result |
| --- | ---: | --- |
| [Base A2V](videos/a2v-controlled-base-holdout.mp4) | none | Rejected: visibly generated face and unstable garment detail. |
| [A2V LoRA, scale 0.50](videos/a2v-controlled-lora-scale-050-holdout.mp4) | 0.50 | Rejected: the adapter changes the result but does not reach photorealism. |
| [A2V LoRA, scale 0.80](videos/a2v-controlled-lora-scale-080-holdout.mp4) | 0.80 | Rejected: still has visible synthetic face and clothing artifacts. |

The controls used `image_strength: 1.0`, `audio_strength: 1.0`, static camera guidance, disabled prompt expansion, and a fixed seed. This means the comparison is not confounded by a different source image, speech track, location, prompt expansion, or random seed.

## What the evidence says

1. The initial prompt-only result was not a valid verdict on this A2V adapter. Fal defines the A2V trainer as a LoRA for a **start image plus conditioning audio**; it is meant to animate that supplied image. [Fal A2V trainer docs](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api)
2. The valid same-input comparison above does show that the adapter affects the output, so the failure is not simply “the LoRA was ignored.”
3. Neither tested adapter scale removes the visible face, skin, hair, and garment-detail artifacts. This blocks acceptance before evaluating broader location coverage or longer duration.
4. The trainer preview is not decisive evidence either way: Fal documents it as a single-stage approximation, while production distilled inference uses a different two-stage path. [Fal A2V trainer docs](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api)

## Root-cause evidence

The input archive is technically consistent: 12 training groups, 5 held-out groups, 544×960, 89 frames, 24 fps, and 48 kHz audio. The problem is not a mismatched frame-rate or malformed archive.

Two substantive limits remain:

- The training-quality attestation marks all 12 training clips as having no clearly visible inner-mouth/teeth frames. LTX’s own character-LoRA guidance calls out real varied clips and inside-of-mouth evidence for lip-sync. [LTX LoRA training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)
- A2V is inherently first-frame conditioned. A strong input-image setting preserves the supplied starting scene; it cannot by itself prove a new arbitrary location while keeping a subject exact. Fal exposes `image_strength` precisely because the input image is a conditioning signal. [Fal custom A2V LoRA API](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora/api)

The remaining budget is not enough to make a convincing, isolated correction to both the training data and the training architecture. Retrying with fewer steps or arbitrary scale changes would not test a well-supported hypothesis, so no additional paid retraining was submitted.

## Spend record

This is an internal conservative reservation record, not a provider invoice.

| Item | Amount (USD) |
| --- | ---: |
| Earlier managed pilot work | 3.5409 |
| Completed 1,000-step A2V training | 6.0000 |
| Earlier native-control reservations | 0.3600 |
| This three-video controlled matrix | 0.3600 |
| **Accounted total** | **10.2609** |
| **Remaining inside the original 12.00 cap** | **1.7391** |

## Production gate

The correct standard is a blind evaluation, not an internal “looks good” judgment. LTX recommends comparing generated and real shots in a randomized review and requiring that reviewers cannot reliably distinguish them. [LTX LoRA training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

This run fails before that gate. It should remain a rejected experiment, with its raw training media, audio, hosted URLs, adapter weights, and private request records kept outside Git. Only the generated diagnostic videos and their hashes are published here.
