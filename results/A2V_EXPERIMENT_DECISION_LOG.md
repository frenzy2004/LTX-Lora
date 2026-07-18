# LTX A2V Experiment Decision Log

**Purpose:** prevent repeating failed routes while pursuing the required standard: generated LTX talking-head footage that cannot be reliably distinguished from genuine footage in blinded playback.

This is a sanitized technical decision record. It contains no raw media, participant names, source paths, provider job URLs, request IDs, adapters, credentials, or chat material.

## Rule for future work

No experiment is retried merely because it is cheap or easy to call. A new provider run must differ in a documented, testable way from a rejected route, and it must preserve a fixed evaluation set so the change can be judged.

| Route | Evidence | Decision | Do not repeat unless |
|---|---|---|---|
| Managed A2V trainer preview | [Published preview](videos/a2v-managed-trainer-preview.mp4) has inverted/negative-looking colour and artifacts. | **Rejected.** | The dataset/preprocessing path changes and Fal's decoded dataset archive is inspected before training. |
| Prompt-only A2V character LoRA | [Published control](videos/a2v-managed-pure-lora-control.mp4) produces a different person. | **Rejected for identity.** | A materially different data/training design is tested; do not reroll the same prompt-only configuration. |
| Reference-anchored A2V | [Published control](videos/a2v-managed-reference-anchor-control.mp4) retains some likeness, but is reference-led and visibly synthetic. | **Not production-ready.** May remain a diagnostic baseline only. | The test measures a defined change in temporal realism and independently reviews normal-speed playback. |
| Reference weakening for a new location | [Published control](videos/a2v-managed-location-tradeoff-control.mp4) did not deliver the requested location change. | **Rejected for location transfer.** | The conditioning/training route changes; do not repeat the same reference-strength trade-off. |
| Full-22B three-location matrix | [Location matrix](A2V_FULL22B_LOCATION_MATRIX.md) contains technically valid candidates but no blinded proof of realism. | **Baseline evidence only; not a quality approval.** | The next comparison adds a genuinely source-isolated hold-out and blinded playback review. |
| 400-step mouth-coverage retrain | [Current evaluation](A2V_MOUTH_COVERAGE_RETRAIN_EVALUATION.md) did not meet the realism bar. Submitted clips were 544×960, 3.708-second derivatives; the set had 2 training clips tagged for visible inner-mouth speech. | **Rejected for the required standard.** | The dataset, crop/resize policy, caption scheme, and validation design change together and are inspected before training. |
| 1,000-step provider-decoded quality rebuild | [Current rebuild](A2V_1000_STEP_QUALITY_REBUILD_EVALUATION.md) used a source/location-separated 13/5 split, explicit speech tags, provider-decoded data inspection, and LTX Quality inference. The wall result improved substantially, but the office result invented prominent text and softened motion detail. | **Rejected for indistinguishability; retain as evidence of conditional improvement.** | Visible text/logos are removed from the data, face-dominant framing is strengthened, provider requests run single-flight, and a blinded playback test is retained. |
| Local full LTX-2 training on this PC | The official local trainer requires Linux/CUDA, local model assets, and typically an 80 GB GPU; this machine had 26.38 GiB free during the audit. | **Not viable on this machine.** | Compute and storage environment change. This is not a Fal quality conclusion. |
| Non-LTX lip-sync or generation model | Outside the defined LTX-only scope. | **Out of scope.** | The project scope itself changes. |

## The only permitted next LTX-only route

A future paid run must be a **new Fal-managed LTX A2V LoRA evaluation**, not a rerun of a rejected configuration. Its private execution record must show all of the following before training:

1. A source-to-training manifest that records the selected original region, the exact crop/resize decision, and train/hold-out separation.
2. A speech-caption audit that preserves the existing plain-language description and adds an explicit A/V speech tag where appropriate.
3. Fal `debug_dataset` output reviewed for decoded video, audio, captions, frame rate, duration, and aspect ratio before committing to full training. [Fal A2V Trainer API](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api)
4. A fixed, source-asset- and location-separated evaluation set where the available footage permits it.
5. A post-run real-versus-generated blind playback test before any claim of indistinguishability. LTX's own guide recommends a blinded comparison rather than relying on a successful API response or selected still frames. [LTX training guide](https://ltx.io/blog/training-your-first-lora-on-ltx)

## Cost status

The published evidence set accounted for **$13.8209** before the controlled rebuild. The rebuild adds a known **$7.466491** for two 100-step data checks, one 1,000-step candidate, and two retrieved Quality outputs, producing a known cumulative subtotal of **$21.287391**. Two completed-compute requests had unretrievable results, so any associated provider billing is not included in that subtotal. This log does not authorize another provider request.

## How to use this log

Before any new run, add a short private change record answering:

```text
Which rejected route does this replace?
What exact data/configuration/evaluation variable changed?
What fixed baseline will it be compared against?
What result would make the experiment stop rather than rerun?
```

If those answers are not concrete, the proposed work is a rerun, not a new experiment.
